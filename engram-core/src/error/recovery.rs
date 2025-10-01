//! Error recovery utilities for graceful degradation
//!
//! Provides utilities for automatic retry, fallback strategies, and partial result handling
//! to ensure system resilience in the face of failures.

use super::{EngramError, RecoveryStrategy, Result};
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

/// Error recovery utilities
pub struct ErrorRecovery;

impl ErrorRecovery {
    /// Execute operation with automatic retry on recoverable errors
    ///
    /// # Errors
    ///
    /// Returns the final error if all retry attempts fail or the provided strategy
    /// does not specify retry behavior.
    pub async fn with_retry<T, F, Fut>(operation: F, strategy: RecoveryStrategy) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let RecoveryStrategy::Retry {
            max_attempts,
            backoff_ms,
        } = strategy
        {
            let mut attempts = 0;
            let mut backoff = Duration::from_millis(backoff_ms);

            loop {
                match operation().await {
                    Ok(result) => return Ok(result),
                    Err(e) if attempts < max_attempts && e.is_recoverable() => {
                        attempts += 1;
                        warn!(
                            "Operation failed (attempt {}/{}), retrying in {:?}: {}",
                            attempts, max_attempts, backoff, e
                        );
                        sleep(backoff).await;
                        backoff *= 2; // Exponential backoff
                    }
                    Err(e) => {
                        error!(
                            "Operation failed permanently after {} attempts: {}",
                            attempts + 1,
                            e
                        );
                        return Err(e);
                    }
                }
            }
        } else {
            operation().await
        }
    }

    /// Execute with fallback on error
    ///
    /// # Errors
    ///
    /// Returns the error from the primary operation when it is non-recoverable or
    /// from the fallback when it also fails.
    pub fn with_fallback<T, F, G>(primary: F, fallback: G) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
        G: FnOnce() -> Result<T>,
    {
        match primary() {
            Ok(result) => Ok(result),
            Err(e) if e.is_recoverable() => {
                warn!("Primary operation failed, using fallback: {}", e);
                fallback()
            }
            Err(e) => Err(e),
        }
    }

    /// Execute with multiple fallback strategies
    ///
    /// # Errors
    ///
    /// Returns the last error produced by the provided strategies or an error
    /// indicating that no strategies were supplied.
    pub fn with_cascading_fallbacks<T>(
        strategies: Vec<Box<dyn FnOnce() -> Result<T>>>,
    ) -> Result<T> {
        let mut last_error = None;

        for (i, strategy) in strategies.into_iter().enumerate() {
            match strategy() {
                Ok(result) => {
                    if i > 0 {
                        info!("Succeeded with fallback strategy {}", i);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    warn!("Strategy {} failed: {}", i, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            EngramError::pattern_match_error(
                "No strategies provided",
                RecoveryStrategy::RequiresIntervention {
                    action: "Add at least one fallback strategy".to_string(),
                },
            )
        }))
    }
}

/// Extension trait for Result types to add recovery methods
pub trait ResultExt<T> {
    /// Apply recovery strategy based on error type
    ///
    /// # Errors
    ///
    /// Returns the original error when the configured recovery strategy cannot
    /// recover from it.
    #[must_use]
    fn or_recover(self) -> Self;
    /// Return partial result if available
    ///
    /// # Errors
    ///
    /// Returns the original error when the recovery strategy does not permit
    /// partial results.
    #[must_use]
    fn or_partial(self, partial: T) -> Self;
    /// Log error with appropriate level
    ///
    /// # Errors
    ///
    /// Returns the original error when the operation fails.
    #[must_use]
    fn log_error(self) -> Self;
    /// Continue without feature on certain error types
    ///
    /// # Errors
    ///
    /// Returns the original error when the recovery strategy does not allow
    /// continuing without the failed feature.
    #[must_use]
    fn or_continue_without_feature(self, default: T) -> Self;
}

impl<T> ResultExt<T> for Result<T> {
    fn or_recover(self) -> Self {
        match self {
            Ok(val) => Ok(val),
            Err(e) => {
                match e.recovery_strategy() {
                    RecoveryStrategy::ContinueWithoutFeature => {
                        warn!("Feature unavailable, continuing: {}", e);
                        // This would need to be specialized per type or require Default
                        Err(e)
                    }
                    _ => Err(e),
                }
            }
        }
    }

    fn or_partial(self, partial: T) -> Self {
        match self {
            Ok(val) => Ok(val),
            Err(e) => match e.recovery_strategy() {
                RecoveryStrategy::PartialResult { description } => {
                    warn!("Returning partial result ({}): {}", description, e);
                    Ok(partial)
                }
                _ => Err(e),
            },
        }
    }

    fn log_error(self) -> Self {
        if let Err(ref error) = self {
            log_recovery_outcome(error);
        }
        self
    }

    fn or_continue_without_feature(self, default: T) -> Self {
        match self {
            Ok(val) => Ok(val),
            Err(e) => match e.recovery_strategy() {
                RecoveryStrategy::ContinueWithoutFeature => {
                    info!("Continuing without feature: {}", e);
                    Ok(default)
                }
                _ => Err(e),
            },
        }
    }
}

fn log_recovery_outcome(error: &EngramError) {
    match error.recovery_strategy() {
        RecoveryStrategy::RequiresIntervention { action } => {
            log_requires_intervention(error, action);
        }
        RecoveryStrategy::Retry { .. } => log_retryable_failure(error),
        _ => log_recoverable_failure(error),
    }
}

fn log_requires_intervention(error: &EngramError, action: &str) {
    error!("Operation failed, manual intervention required: {}", error);
    error!("Required action: {}", action);
}

fn log_retryable_failure(error: &EngramError) {
    warn!("Retryable operation failed: {}", error);
}

fn log_recoverable_failure(error: &EngramError) {
    info!("Operation failed but recoverable: {}", error);
}

/// Macro to help migrate unwrap calls gradually
#[macro_export]
macro_rules! try_unwrap {
    ($expr:expr, $recovery:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                tracing::error!("Unwrap failed: {}", e);
                return Err($crate::error::EngramError::memory_error(
                    "Unwrap failed",
                    e,
                    $recovery,
                ));
            }
        }
    };
}

/// Macro to replace expect with proper error handling
#[macro_export]
macro_rules! try_expect {
    ($expr:expr, $msg:literal) => {
        match $expr {
            Ok(val) => val,
            Err(e) => {
                tracing::error!("{}: {}", $msg, e);
                return Err($crate::error::EngramError::memory_error(
                    $msg,
                    e,
                    $crate::error::RecoveryStrategy::RequiresIntervention {
                        action: $msg.to_string(),
                    },
                ));
            }
        }
    };
}

/// Helper function to replace panic! in match arms
///
/// # Errors
///
/// Always returns an error describing the unexpected pattern.
pub fn unreachable_pattern<T>(pattern: &str) -> Result<T> {
    Err(EngramError::pattern_match_error(
        format!("Unexpected pattern: {pattern}"),
        RecoveryStrategy::RequiresIntervention {
            action: format!("Fix pattern matching for: {pattern}"),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    type TestResult<T = ()> = std::result::Result<T, String>;

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    #[tokio::test]
    async fn test_retry_on_transient_failure() -> TestResult {
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let result = ErrorRecovery::with_retry(
            || async {
                let current = attempts_clone.fetch_add(1, Ordering::SeqCst);
                if current < 2 {
                    Err(EngramError::storage_error(
                        "read",
                        std::io::Error::new(std::io::ErrorKind::Interrupted, ""),
                        RecoveryStrategy::Retry {
                            max_attempts: 3,
                            backoff_ms: 10,
                        },
                    ))
                } else {
                    Ok(42)
                }
            },
            RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff_ms: 10,
            },
        )
        .await;

        let value = result.map_err(|err| format!("retry should eventually succeed: {err:?}"))?;
        ensure_eq(&value, &42, "retry result value")?;
        ensure_eq(&attempts.load(Ordering::SeqCst), &3, "retry attempt count")?;
        Ok(())
    }

    #[test]
    fn test_fallback_on_index_failure() -> TestResult {
        let result = ErrorRecovery::with_fallback(
            || {
                Err(EngramError::Index {
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "Index file missing",
                    )),
                    fallback_available: true,
                    recovery: RecoveryStrategy::Fallback {
                        description: "Use linear search".to_string(),
                    },
                })
            },
            || {
                // Fallback to linear search
                Ok(vec![1, 2, 3])
            },
        );

        let values = result.map_err(|err| format!("fallback should succeed: {err:?}"))?;
        ensure_eq(&values, &vec![1, 2, 3], "fallback return value")
    }

    #[test]
    fn test_cascading_fallbacks() -> TestResult {
        let strategies: Vec<Box<dyn FnOnce() -> Result<i32>>> = vec![
            Box::new(|| {
                Err(EngramError::storage_error(
                    "primary failed",
                    std::io::Error::new(std::io::ErrorKind::NotFound, ""),
                    RecoveryStrategy::Fallback {
                        description: "Try fallback".to_string(),
                    },
                ))
            }),
            Box::new(|| {
                Err(EngramError::storage_error(
                    "fallback 1 failed",
                    std::io::Error::new(std::io::ErrorKind::PermissionDenied, ""),
                    RecoveryStrategy::Fallback {
                        description: "Try next fallback".to_string(),
                    },
                ))
            }),
            Box::new(|| Ok(42)),
        ];

        let result = ErrorRecovery::with_cascading_fallbacks(strategies);
        let value = result.map_err(|err| format!("cascading fallback should recover: {err:?}"))?;
        ensure_eq(&value, &42, "cascading fallback result")
    }

    #[test]
    fn test_partial_results_extension() -> TestResult {
        use crate::Memory;

        let partial_memories = vec![Memory::new(
            "partial1".to_string(),
            [0.1; 768],
            crate::Confidence::LOW,
        )];

        let result: Result<Vec<Memory>> = Err(EngramError::query_error(
            "Timeout during search",
            Some(partial_memories.clone()),
            RecoveryStrategy::PartialResult {
                description: "Returning results found before timeout".to_string(),
            },
        ));

        let recovered = result
            .or_partial(partial_memories)
            .map_err(|err| format!("partial results should be returned: {err:?}"))?;
        ensure_eq(&recovered.len(), &1_usize, "partial result count")
    }

    #[test]
    fn test_continue_without_feature() -> TestResult {
        let result: Result<Vec<i32>> = Err(EngramError::Index {
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Feature unavailable",
            )),
            fallback_available: false,
            recovery: RecoveryStrategy::ContinueWithoutFeature,
        });

        let default_value = vec![1, 2, 3];
        let recovered = result
            .or_continue_without_feature(default_value.clone())
            .map_err(|err| format!("should continue without feature: {err:?}"))?;
        ensure_eq(
            &recovered,
            &default_value,
            "continue without feature result",
        )
    }
}
