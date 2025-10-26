const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library for FFI - exports performance kernels to Rust
    const lib = b.addStaticLibrary(.{
        .name = "engram_kernels",
        .root_source_file = b.path("src/ffi.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Enable C ABI exports and bundle compiler runtime
    lib.bundle_compiler_rt = true;

    // Install the static library to zig-out/lib/
    b.installArtifact(lib);

    // Unit tests for Zig kernels
    const ffi_tests = b.addTest(.{
        .root_source_file = b.path("src/ffi.zig"),
        .target = target,
        .optimize = optimize,
    });

    const allocator_tests = b.addTest(.{
        .root_source_file = b.path("src/allocator_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    const vector_similarity_tests = b.addTest(.{
        .root_source_file = b.path("src/vector_similarity_test.zig"),
        .target = target,
        .optimize = optimize,
    });

    const spreading_activation_tests = b.addTest(.{
        .root_source_file = b.path("src/spreading_activation.zig"),
        .target = target,
        .optimize = optimize,
    });

    const decay_functions_tests = b.addTest(.{
        .root_source_file = b.path("src/decay_functions.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_ffi_tests = b.addRunArtifact(ffi_tests);
    const run_allocator_tests = b.addRunArtifact(allocator_tests);
    const run_vector_similarity_tests = b.addRunArtifact(vector_similarity_tests);
    const run_spreading_activation_tests = b.addRunArtifact(spreading_activation_tests);
    const run_decay_functions_tests = b.addRunArtifact(decay_functions_tests);

    const test_step = b.step("test", "Run Zig kernel unit tests");
    test_step.dependOn(&run_ffi_tests.step);
    test_step.dependOn(&run_allocator_tests.step);
    test_step.dependOn(&run_vector_similarity_tests.step);
    test_step.dependOn(&run_spreading_activation_tests.step);
    test_step.dependOn(&run_decay_functions_tests.step);
}
