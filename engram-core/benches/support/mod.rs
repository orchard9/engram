pub mod ann_common;
pub mod datasets;
pub mod engram_ann;

// Real ANN library implementations (require feature flag and system libraries)
#[cfg(feature = "ann_benchmarks")]
pub mod annoy_ann;
#[cfg(feature = "ann_benchmarks")]
pub mod faiss_ann; // Note: Uses mock impl due to annoy-rs API limitations
