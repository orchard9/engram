//! Integration tests for error recovery systems
//!
//! Tests the complete error handling and recovery pipeline from EngramError
//! through recovery strategies to graceful degradation.

use engram_core::error::{EngramError, RecoveryStrategy, ErrorRecovery, ResultExt};
use std::error::Error;
use engram_core::{Memory, Confidence};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::timeout;
use std::time::Duration;

#[tokio::test]
async fn test_retry_strategy_with_exponential_backoff() {
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    
    let start = std::time::Instant::now();
    
    let result = ErrorRecovery::with_retry(
        || async {
            let current = attempts_clone.fetch_add(1, Ordering::SeqCst);
            if current < 2 {
                Err(EngramError::storage_error(
                    "simulated_failure",
                    std::io::Error::new(std::io::ErrorKind::Interrupted, ""),
                    RecoveryStrategy::Retry {
                        max_attempts: 3,
                        backoff_ms: 50,
                    },
                ))
            } else {
                Ok("success")
            }
        },
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 50,
        },
    ).await;
    
    let duration = start.elapsed();
    
    assert_eq!(result.unwrap(), "success");
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    
    // Should have exponential backoff: 50ms + 100ms = 150ms minimum
    assert!(duration >= Duration::from_millis(150));
    assert!(duration < Duration::from_millis(500)); // But not too long
}

#[tokio::test]
async fn test_retry_strategy_permanent_failure() {
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    
    let result: Result<&str, EngramError> = ErrorRecovery::with_retry(
        || async {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            Err(EngramError::storage_error(
                "permanent_failure",
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, ""),
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    backoff_ms: 10,
                },
            ))
        },
        RecoveryStrategy::Retry {
            max_attempts: 2,
            backoff_ms: 10,
        },
    ).await;
    
    assert!(result.is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), 3); // 2 retries + original attempt
    
    let error = result.unwrap_err();
    assert!(matches!(error, EngramError::Storage { .. }));
    // Check recovery strategy by pattern matching since RecoveryStrategy doesn't implement PartialEq
    match error.recovery_strategy() {
        RecoveryStrategy::Retry { max_attempts, backoff_ms } => {
            assert_eq!(*max_attempts, 2);
            assert_eq!(*backoff_ms, 10);
        },
        _ => panic!("Expected Retry recovery strategy"),
    }
}

#[test]
fn test_fallback_strategy_success() {
    let result = ErrorRecovery::with_fallback(
        || {
            Err(EngramError::Index {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "HNSW index not available"
                )),
                fallback_available: true,
                recovery: RecoveryStrategy::Fallback {
                    description: "Use linear search".to_string(),
                },
            })
        },
        || {
            // Fallback to linear search
            Ok(vec!["linear_result_1", "linear_result_2"])
        },
    );
    
    assert_eq!(result.unwrap(), vec!["linear_result_1", "linear_result_2"]);
}

#[test]
fn test_cascading_fallbacks() {
    let strategies: Vec<Box<dyn FnOnce() -> Result<String, EngramError>>> = vec![
        // Primary strategy fails
        Box::new(|| {
            Err(EngramError::storage_error(
                "primary_storage_failure",
                std::io::Error::new(std::io::ErrorKind::NotFound, ""),
                RecoveryStrategy::Fallback {
                    description: "Try backup storage".to_string(),
                },
            ))
        }),
        // First fallback fails
        Box::new(|| {
            Err(EngramError::storage_error(
                "backup_storage_failure",
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, ""),
                RecoveryStrategy::Fallback {
                    description: "Try cache".to_string(),
                },
            ))
        }),
        // Cache succeeds
        Box::new(|| Ok("cache_result".to_string())),
    ];
    
    let result = ErrorRecovery::with_cascading_fallbacks(strategies);
    assert_eq!(result.unwrap(), "cache_result");
}

#[test]
fn test_partial_results_recovery() {
    let partial_memories = vec![
        Memory::new("partial1".to_string(), [0.1; 768], Confidence::LOW),
        Memory::new("partial2".to_string(), [0.2; 768], Confidence::MEDIUM),
    ];
    
    let result: Result<Vec<Memory>, EngramError> = Err(EngramError::query_error(
        "Search timeout after 5 seconds",
        Some(partial_memories.clone()),
        RecoveryStrategy::PartialResult {
            description: "Returning memories found before timeout".to_string(),
        },
    ));
    
    let recovered = result.or_partial(partial_memories.clone());
    assert!(recovered.is_ok());
    assert_eq!(recovered.unwrap().len(), 2);
}

#[test]
fn test_continue_without_feature_recovery() {
    let result: Result<Vec<String>, EngramError> = Err(EngramError::Index {
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HNSW feature not compiled"
        )),
        fallback_available: false,
        recovery: RecoveryStrategy::ContinueWithoutFeature,
    });
    
    let default_results = vec!["default_result".to_string()];
    let recovered = result.or_continue_without_feature(default_results.clone());
    assert_eq!(recovered.unwrap(), default_results);
}

#[test]
fn test_error_logging_levels() {
    // Test different recovery strategies produce appropriate log levels
    let critical_error: Result<(), EngramError> = Err(EngramError::WriteAheadLog {
        operation: "critical_write".to_string(),
        source: std::io::Error::new(std::io::ErrorKind::Other, "disk failure"),
        recovery: RecoveryStrategy::RequiresIntervention {
            action: "Replace failed disk".to_string(),
        },
        can_continue: false,
    });
    
    let logged_result = critical_error.log_error();
    assert!(logged_result.is_err());
    
    let retriable_error: Result<(), EngramError> = Err(EngramError::storage_error(
        "transient_failure",
        std::io::Error::new(std::io::ErrorKind::Interrupted, ""),
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 100,
        },
    ));
    
    let logged_result = retriable_error.log_error();
    assert!(logged_result.is_err());
}

#[test]
fn test_error_recoverability_classification() {
    let recoverable_error = EngramError::storage_error(
        "temp_failure",
        std::io::Error::new(std::io::ErrorKind::TimedOut, ""),
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 100,
        },
    );
    assert!(recoverable_error.is_recoverable());
    
    let non_recoverable_error = EngramError::WriteAheadLog {
        operation: "write_critical_data".to_string(),
        source: std::io::Error::new(std::io::ErrorKind::Other, "corruption detected"),
        recovery: RecoveryStrategy::RequiresIntervention {
            action: "Manual data recovery required".to_string(),
        },
        can_continue: false,
    };
    assert!(!non_recoverable_error.is_recoverable());
}

#[test]
fn test_evidence_type_mismatch_recovery() {
    let error = EngramError::evidence_type_mismatch(
        "DirectMatch",
        "SpreadingActivation",
        RecoveryStrategy::PartialResult {
            description: "Skip incompatible evidence".to_string(),
        },
    );
    
    assert!(error.is_recoverable());
    assert!(error.to_string().contains("expected DirectMatch"));
    assert!(error.to_string().contains("got SpreadingActivation"));
}

#[test]
fn test_cue_type_mismatch_recovery() {
    let error = EngramError::cue_type_mismatch(
        "Embedding",
        "Semantic",
        RecoveryStrategy::Fallback {
            description: "Convert to compatible cue type".to_string(),
        },
    );
    
    assert!(error.is_recoverable());
    match error.recovery_strategy() {
        RecoveryStrategy::Fallback { description } => {
            assert_eq!(description, "Convert to compatible cue type");
        }
        _ => panic!("Wrong recovery strategy"),
    }
}

#[test]
fn test_pattern_match_error_recovery() {
    let error = EngramError::pattern_match_error(
        "Unknown cue type variant",
        RecoveryStrategy::RequiresIntervention {
            action: "Update pattern matching to handle new variants".to_string(),
        },
    );
    
    assert!(!error.is_recoverable());
    assert!(error.to_string().contains("Pattern matching failed"));
}

#[tokio::test]
async fn test_timeout_with_recovery() {
    let slow_operation = async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok::<&str, EngramError>("completed")
    };
    
    // Test that recovery works even with timeouts
    let result = timeout(Duration::from_millis(100), slow_operation).await;
    assert!(result.is_err()); // Should timeout
    
    // Now test with retry after timeout
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    
    let result = ErrorRecovery::with_retry(
        || async {
            let current = attempts_clone.fetch_add(1, Ordering::SeqCst);
            if current < 2 {
                Err(EngramError::query_error(
                    "Query timeout",
                    None,
                    RecoveryStrategy::Retry {
                        max_attempts: 3,
                        backoff_ms: 10,
                    },
                ))
            } else {
                Ok("success_after_retry")
            }
        },
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 10,
        },
    ).await;
    
    assert_eq!(result.unwrap(), "success_after_retry");
}

#[test]
fn test_memory_error_creation() {
    let source_error = std::io::Error::new(std::io::ErrorKind::OutOfMemory, "allocation failed");
    
    let error = EngramError::memory_error(
        "Failed to allocate memory for episode storage",
        source_error,
        RecoveryStrategy::Fallback {
            description: "Use disk-based storage".to_string(),
        },
    );
    
    assert!(error.is_recoverable());
    assert!(error.to_string().contains("Memory operation failed"));
    assert!(error.source().is_some());
}

#[test]
fn test_query_error_with_partial_results() {
    let partial_results = vec![
        Memory::new("result1".to_string(), [0.1; 768], Confidence::HIGH),
        Memory::new("result2".to_string(), [0.2; 768], Confidence::MEDIUM),
    ];
    
    let error = EngramError::query_error(
        "Search incomplete due to index corruption",
        Some(partial_results.clone()),
        RecoveryStrategy::PartialResult {
            description: "Return available results".to_string(),
        },
    );
    
    if let EngramError::Query { partial_results: Some(ref results), .. } = error {
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "result1");
    } else {
        panic!("Expected Query error with partial results");
    }
}

/// Test that our error recovery doesn't create infinite loops
#[tokio::test]
async fn test_no_infinite_retry_loops() {
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts_clone = attempts.clone();
    
    let start = std::time::Instant::now();
    
    let result: Result<&str, EngramError> = ErrorRecovery::with_retry(
        || async {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            Err(EngramError::storage_error(
                "always_fails",
                std::io::Error::new(std::io::ErrorKind::Other, ""),
                RecoveryStrategy::Retry {
                    max_attempts: 3,
                    backoff_ms: 10,
                },
            ))
        },
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 10,
        },
    ).await;
    
    let duration = start.elapsed();
    
    assert!(result.is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), 4); // 3 retries + 1 original
    
    // Should not take too long even with retries
    assert!(duration < Duration::from_secs(1));
}

/// Test error recovery under concurrent load
#[tokio::test]
async fn test_concurrent_error_recovery() {
    let tasks = (0..10).map(|i| {
        tokio::spawn(async move {
            let result = ErrorRecovery::with_retry(
                || async {
                    if i % 2 == 0 {
                        Ok(format!("success_{}", i))
                    } else {
                        Err(EngramError::storage_error(
                            "intermittent_failure",
                            std::io::Error::new(std::io::ErrorKind::Interrupted, ""),
                            RecoveryStrategy::Retry {
                                max_attempts: 2,
                                backoff_ms: 5,
                            },
                        ))
                    }
                },
                RecoveryStrategy::Retry {
                    max_attempts: 2,
                    backoff_ms: 5,
                },
            ).await;
            
            (i, result)
        })
    }).collect::<Vec<_>>();
    
    let results = futures::future::join_all(tasks).await;
    
    // Even-numbered tasks should succeed
    for result in results {
        let (i, task_result) = result.unwrap();
        if i % 2 == 0 {
            assert!(task_result.is_ok());
            assert_eq!(task_result.unwrap(), format!("success_{}", i));
        } else {
            assert!(task_result.is_err());
        }
    }
}