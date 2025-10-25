# Twitter Thread: CUDA-Cargo Build Integration

## Tweet 1 (Hook)
Building Rust + CUDA is where GPU projects die.

Rust wants .rs files. CUDA wants .cu files. Your CI wants CPU-only builds.

Thread on making CUDA and Cargo actually work together in production:

## Tweet 2 (The Build Script)
Cargo's build.rs is the integration layer:

1. Detect CUDA availability
2. Compile .cu kernels with nvcc
3. Generate FFI bindings
4. Link CUDA runtime
5. Set conditional compilation flags

If CUDA missing: gracefully fall back to CPU-only build.

## Tweet 3 (Fat Binaries)
One binary, multiple GPU architectures:

```bash
nvcc -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_86,code=sm_86 \
     kernel.cu
```

Runtime auto-selects correct code.

Cost: 4x binary size
Benefit: works on GTX 1060 through H100

## Tweet 4 (PTX vs SASS Trade-Off)
Alternative: include PTX, JIT-compile at runtime

PTX pros: smaller binary, forward compatible
PTX cons: 100-500ms first-run delay

We choose fat binaries. User experience beats binary size.

## Tweet 5 (Type-Safe FFI)
Manual FFI is UB-prone. One wrong type = crashes.

Bindgen auto-generates correct bindings from C headers:

```rust
bindgen::Builder::default()
    .header("cuda/api.h")
    .generate()
```

Type mismatches become compile errors, not runtime crashes.

## Tweet 6 (Conditional Compilation)
Make GPU optional:

```rust
#[cfg(cuda_enabled)]
mod cuda_impl;

#[cfg(cpu_only)]
mod cpu_impl;
```

User code is backend-agnostic. GPU is an implementation detail.

Builds succeed with or without CUDA toolkit.

## Tweet 7 (Build Caching)
CUDA compilation is slow (10-30s).

Cargo caching saves iteration time:

```rust
println!("cargo:rerun-if-changed=cuda/kernel.cu");
```

Only rebuild CUDA when kernels change.

Result: 2s builds during development, not 20s.

## Tweet 8 (Call to Action)
Build systems aren't glamorous. But they're foundational.

Broken build = nobody can compile
Slow build = productivity tanks

Get the build system right first. Everything else builds on it.

Building production GPU systems: https://github.com/YourOrg/engram
