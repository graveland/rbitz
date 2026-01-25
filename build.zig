const std = @import("std");

fn getVersion(b: *std.Build) []const u8 {
    const src_dir = std.fs.path.dirname(@src().file) orelse ".";
    var exit_code: u8 = 0;
    const git_hash = b.runAllowFail(&[_][]const u8{
        "git", "-C", src_dir, "rev-parse", "HEAD",
    }, &exit_code, .inherit) catch return "unknown";
    return std.mem.trim(u8, git_hash, &std.ascii.whitespace);
}

// CRoaring source files relative to src/
const croaring_source_files = [_][]const u8{
    "array_util.c",
    "art/art.c",
    "bitset_util.c",
    "bitset.c",
    "containers/array.c",
    "containers/bitset.c",
    "containers/containers.c",
    "containers/convert.c",
    "containers/mixed_andnot.c",
    "containers/mixed_equal.c",
    "containers/mixed_intersection.c",
    "containers/mixed_negation.c",
    "containers/mixed_subset.c",
    "containers/mixed_union.c",
    "containers/mixed_xor.c",
    "containers/run.c",
    "isadetection.c",
    "memory.c",
    "roaring_array.c",
    "roaring_priority_queue.c",
    "roaring.c",
    "roaring64.c",
};

const croaring_flags = [_][]const u8{
    "-std=c11",
    "-O3",
    "-fno-sanitize=undefined",
    "-DROARING_EXCEPTIONS=1",
    "-DCROARING_COMPILER_SUPPORTS_AVX512=0",
};

/// Creates the rbitz module using CRoaring from zig dependency.
/// Use this when incorporating rbitz as a dependency.
pub fn createModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_source_file: std.Build.LazyPath,
) *std.Build.Module {
    // Get CRoaring from zig dependency
    const croaring_dep = b.dependency("croaring", .{});

    // Create the rbitz module
    const mod = b.addModule("rbitz", .{
        .root_source_file = root_source_file,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Add CRoaring include path
    mod.addIncludePath(croaring_dep.path("include"));

    // Add CRoaring C sources
    mod.addCSourceFiles(.{
        .root = croaring_dep.path("src"),
        .files = &croaring_source_files,
        .flags = &croaring_flags,
    });

    // Add version info
    const options = b.addOptions();
    options.addOption([]const u8, "version", getVersion(b));
    mod.addOptions("build_options", options);

    return mod;
}

/// Standalone build function for building rbitz directly.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get CRoaring from zig dependency
    const croaring_dep = b.dependency("croaring", .{});

    // Create the rbitz module
    const mod = b.addModule("rbitz", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Add CRoaring include path
    mod.addIncludePath(croaring_dep.path("include"));

    // Add CRoaring C sources
    mod.addCSourceFiles(.{
        .root = croaring_dep.path("src"),
        .files = &croaring_source_files,
        .flags = &croaring_flags,
    });

    // Add version info
    const options = b.addOptions();
    options.addOption([]const u8, "version", getVersion(b));
    mod.addOptions("build_options", options);

    // Build executable - just imports the module, no need to re-add CRoaring
    const exe = b.addExecutable(.{
        .name = "rbitz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "rbitz", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Tests
    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Benchmarks (always ReleaseFast for accurate measurements)
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "rbitz", .module = mod },
            },
        }),
    });
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    run_bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bench.addArgs(args);
    }

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
