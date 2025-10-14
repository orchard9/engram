#![allow(clippy::all)]
#![allow(missing_docs)]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../scripts/fuzz_spreading_latency.rs"
));
