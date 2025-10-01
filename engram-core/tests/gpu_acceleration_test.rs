//! Integration tests for GPU acceleration foundation

use engram_core::activation::{
    create_activation_graph, ActivationGraphExt, AdaptiveConfig, AdaptiveSpreadingEngine,
    CpuFallback, EdgeType, GPUActivationBatch, GPUSpreadingInterface, MockGpuInterface,
    ParallelSpreadingConfig, ParallelSpreadingEngine,
};
use std::sync::Arc;

#[test]
fn test_cpu_fallback_processing() {
    let fallback = CpuFallback::new();
    assert!(fallback.is_available());

    let mut batch = GPUActivationBatch::new([1.0; 768]);
    batch.add_target([1.0; 768], 0.5, 0.6); // Same vector, similarity = 1.0
    batch.add_target([0.0; 768], 0.3, 0.4); // Zero vector, similarity = 0.0

    let results = fallback.launch(&batch);
    // In synchronous context, the future is immediately ready
    assert!(futures::executor::block_on(results).is_ok());
}

#[test]
fn test_adaptive_engine_dispatch() {
    // Test with GPU disabled
    let config_cpu = AdaptiveConfig {
        gpu_threshold: 10,
        enable_gpu: false,
    };
    let mut engine_cpu = AdaptiveSpreadingEngine::new(None, config_cpu);

    let mut batch = GPUActivationBatch::new([1.0; 768]);
    for i in 0..20 {
        let mut target = [0.0; 768];
        target[i] = 1.0;
        batch.add_target(target, 0.5, 0.5);
    }

    // Should use CPU even though batch size exceeds threshold
    let result = futures::executor::block_on(engine_cpu.spread(&batch));
    assert!(result.is_ok());
    assert_eq!(engine_cpu.backend_name(), "CPU_SIMD_FALLBACK");
}

#[test]
fn test_adaptive_engine_with_mock_gpu() {
    let mock_gpu = Arc::new(MockGpuInterface::new(true));
    let config = AdaptiveConfig {
        gpu_threshold: 5,
        enable_gpu: true,
    };
    let mut engine = AdaptiveSpreadingEngine::new(Some(mock_gpu), config);

    let mut batch = GPUActivationBatch::new([1.0; 768]);
    for i in 0..10 {
        let mut target = [0.0; 768];
        target[i] = 1.0;
        batch.add_target(target, 0.5, 0.5);
    }

    // Should use GPU since batch size exceeds threshold and GPU is enabled
    let result = futures::executor::block_on(engine.spread(&batch));
    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 10);
    // Mock GPU returns all 1.0 values
    assert!(values.iter().all(|&v| v == 1.0));
}

#[test]
fn test_parallel_engine_with_gpu_config() {
    let config = ParallelSpreadingConfig {
        num_threads: 2,
        max_depth: 2,
        enable_gpu: false, // Disabled by default
        gpu_threshold: 64,
        ..Default::default()
    };

    let graph = create_activation_graph();
    ActivationGraphExt::add_edge(&graph, "A".to_string(), "B".to_string(), 0.8, EdgeType::Excitatory);
    ActivationGraphExt::add_edge(&graph, "B".to_string(), "C".to_string(), 0.6, EdgeType::Excitatory);

    let engine = ParallelSpreadingEngine::new(config, Arc::new(graph)).unwrap();

    // Spread activation from node A
    let seed_activations = vec![("A".to_string(), 1.0)];
    let results = engine.spread_activation(&seed_activations).unwrap();

    assert!(!results.activations.is_empty());
    engine.shutdown().unwrap();
}

#[test]
fn test_batch_memory_layout() {
    let mut batch = GPUActivationBatch::new([1.0; 768]);
    assert!(batch.is_empty());

    // Reserve capacity to ensure contiguous allocation
    batch.reserve(100);

    // Add targets
    for i in 0..50 {
        let mut target = [0.0; 768];
        target[i % 768] = 1.0;
        batch.add_target(target, 0.1 * i as f32, 0.2 * i as f32);
    }

    assert_eq!(batch.size(), 50);
    batch.ensure_contiguous();

    // Verify data integrity
    assert_eq!(batch.activations.len(), 50);
    assert_eq!(batch.confidences.len(), 50);
    assert_eq!(batch.targets.len(), 50);
}

#[tokio::test]
async fn test_async_gpu_interface() {
    let fallback = CpuFallback::new();

    let mut batch = GPUActivationBatch::new([1.0; 768]);
    batch.add_target([1.0; 768], 1.0, 1.0);

    let future = fallback.launch(&batch);
    let result = future.await;

    assert!(result.is_ok());
    let values = result.unwrap();
    assert_eq!(values.len(), 1);
}

#[test]
fn test_gpu_config_propagation() {
    let config = ParallelSpreadingConfig {
        enable_gpu: true,
        gpu_threshold: 32,
        ..Default::default()
    };

    assert!(config.enable_gpu);
    assert_eq!(config.gpu_threshold, 32);

    // Verify defaults
    let default_config = ParallelSpreadingConfig::default();
    assert!(!default_config.enable_gpu);
    assert_eq!(default_config.gpu_threshold, 64);
}