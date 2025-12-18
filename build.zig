const std = @import("std");

// Get the directory where this build.zig lives
fn getSrcDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

fn getVersion(b: *std.Build) []const u8 {
    const src_dir = getSrcDir();
    var exit_code: u8 = 0;
    const git_hash = b.runAllowFail(&[_][]const u8{
        "git", "-C", src_dir, "rev-parse", "HEAD",
    }, &exit_code, .Inherit) catch return "unknown";
    return std.mem.trim(u8, git_hash, &std.ascii.whitespace);
}

fn checkLocalCRoaringVersionAt(b: *std.Build, base_dir: []const u8) !void {
    const header_path = b.pathJoin(&.{ base_dir, "CRoaring/include/roaring/roaring_version.h" });
    const content = try std.fs.cwd().readFileAlloc(
        header_path,
        b.allocator,
        .limited(1024 * 1024),
    );
    defer b.allocator.free(content);

    try checkVersionContent(content);
}

fn checkVersionContent(content: []const u8) !void {
    var major: ?u32 = null;
    var minor: ?u32 = null;

    var lines = std.mem.tokenizeScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "ROARING_VERSION_MAJOR")) |_| {
            var parts = std.mem.tokenizeAny(u8, line, " =,");
            while (parts.next()) |part| {
                if (std.fmt.parseInt(u32, part, 10)) |value| {
                    major = value;
                    break;
                } else |_| {}
            }
        } else if (std.mem.indexOf(u8, line, "ROARING_VERSION_MINOR")) |_| {
            var parts = std.mem.tokenizeAny(u8, line, " =,");
            while (parts.next()) |part| {
                if (std.fmt.parseInt(u32, part, 10)) |value| {
                    minor = value;
                    break;
                } else |_| {}
            }
        }
    }

    const major_ver = major orelse return error.VersionNotFound;
    const minor_ver = minor orelse return error.VersionNotFound;

    if (major_ver < 4 or (major_ver == 4 and minor_ver < 4)) {
        return error.VersionTooOld;
    }
}

fn hasLocalCRoaringAt(b: *std.Build, base_dir: []const u8) bool {
    const src_path = b.pathJoin(&.{ base_dir, "CRoaring/src/roaring.c" });
    std.fs.cwd().access(src_path, .{}) catch return false;
    return true;
}

const SystemRoaring = struct {
    include_path: []const u8,
    lib_path: ?[]const u8,
};

fn findSystemRoaring(b: *std.Build) ?SystemRoaring {
    const result = std.process.Child.run(.{
        .allocator = b.allocator,
        .argv = &.{ "pkg-config", "--cflags", "--libs", "roaring" },
    }) catch return null;

    if (result.term.Exited != 0) return null;

    var include_path: ?[]const u8 = null;
    var lib_path: ?[]const u8 = null;

    var parts = std.mem.tokenizeScalar(u8, result.stdout, ' ');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "-I")) {
            include_path = trimmed[2..];
        } else if (std.mem.startsWith(u8, trimmed, "-L")) {
            lib_path = trimmed[2..];
        }
    }

    if (include_path) |inc| {
        // Try to verify the version by reading the header
        const version_path = std.fs.path.join(b.allocator, &.{ inc, "roaring/roaring_version.h" }) catch return null;
        const content = std.fs.cwd().readFileAlloc(version_path, b.allocator, .limited(1024 * 1024)) catch return null;
        checkVersionContent(content) catch {
            std.debug.print("WARNING: System roaring library version is too old (requires >= 4.4.0)\n", .{});
            return null;
        };
        return .{ .include_path = inc, .lib_path = lib_path };
    }

    return null;
}

// CRoaring source files relative to CRoaring directory
const croaring_source_files = [_][]const u8{
    "src/array_util.c",
    "src/art/art.c",
    "src/bitset_util.c",
    "src/bitset.c",
    "src/containers/array.c",
    "src/containers/bitset.c",
    "src/containers/containers.c",
    "src/containers/convert.c",
    "src/containers/mixed_andnot.c",
    "src/containers/mixed_equal.c",
    "src/containers/mixed_intersection.c",
    "src/containers/mixed_negation.c",
    "src/containers/mixed_subset.c",
    "src/containers/mixed_union.c",
    "src/containers/mixed_xor.c",
    "src/containers/run.c",
    "src/isadetection.c",
    "src/memory.c",
    "src/roaring_array.c",
    "src/roaring_priority_queue.c",
    "src/roaring.c",
    "src/roaring64.c",
};

/// CRoaring setup result
const CRoaringInfo = struct {
    lib: ?*std.Build.Step.Compile,
    include_path: std.Build.LazyPath,
};

/// Internal helper to set up CRoaring (local submodule or system library).
/// Used by both createModule() and build().
fn setupCRoaring(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    base_dir: []const u8,
) CRoaringInfo {
    // Determine whether to use local CRoaring source or system library
    const use_local = hasLocalCRoaringAt(b, base_dir);
    const system_roaring = if (!use_local) findSystemRoaring(b) else null;

    if (!use_local and system_roaring == null) {
        std.debug.print("ERROR: CRoaring not found.\n", .{});
        std.debug.print("Either:\n", .{});
        std.debug.print("  1. Add CRoaring as a submodule in CRoaring/\n", .{});
        std.debug.print("  2. Install roaring library system-wide (with pkg-config support)\n", .{});
        std.debug.print("Required version: >= 4.4.0\n", .{});
        std.process.exit(1);
    }

    if (use_local) {
        checkLocalCRoaringVersionAt(b, base_dir) catch |err| {
            std.debug.print("ERROR: CRoaring version check failed: {}\n", .{err});
            std.debug.print("Required version: >= 4.4.0\n", .{});
            std.process.exit(1);
        };

        // Create a module for CRoaring (C-only, no root source file)
        const croaring_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });

        const croaring_include_path = b.pathJoin(&.{ base_dir, "CRoaring/include" });
        croaring_module.addIncludePath(.{ .cwd_relative = croaring_include_path });

        // Add all CRoaring C source files with full paths
        var sources: [croaring_source_files.len][]const u8 = undefined;
        for (croaring_source_files, 0..) |src, i| {
            sources[i] = b.pathJoin(&.{ base_dir, "CRoaring", src });
        }

        croaring_module.addCSourceFiles(.{
            .files = &sources,
            .flags = &[_][]const u8{
                "-std=c11",
                "-O3",
                "-fno-sanitize=undefined",
                "-DROARING_EXCEPTIONS=1",
                "-DCROARING_COMPILER_SUPPORTS_AVX512=0",
            },
        });

        // Create the static library using the module
        const lib = b.addLibrary(.{
            .name = "roaring",
            .root_module = croaring_module,
            .linkage = .static,
        });
        return .{
            .lib = lib,
            .include_path = .{ .cwd_relative = croaring_include_path },
        };
    } else {
        // Using system library
        const sys = system_roaring.?;
        return .{
            .lib = null,
            .include_path = .{ .cwd_relative = sys.include_path },
        };
    }
}

/// Helper to link CRoaring to a module
fn linkCRoaring(mod: *std.Build.Module, info: CRoaringInfo) void {
    mod.addIncludePath(info.include_path);
    if (info.lib) |lib| {
        mod.linkLibrary(lib);
    } else {
        mod.linkSystemLibrary("roaring", .{});
    }
}

/// Creates the rbitz module with CRoaring handling.
/// Use this when incorporating rbitz as a dependency to share modules with parent.
/// The function handles detection and building of CRoaring (local submodule or system library).
pub fn createModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_source_file: std.Build.LazyPath,
) *std.Build.Module {
    const src_dir = getSrcDir();
    const croaring = setupCRoaring(b, target, optimize, src_dir);

    // Create the rbitz module
    const mod = b.addModule("rbitz", .{
        .root_source_file = root_source_file,
        .target = target,
        .optimize = optimize,
    });

    linkCRoaring(mod, croaring);

    // Add version info
    const options = b.addOptions();
    options.addOption([]const u8, "version", getVersion(b));
    mod.addOptions("build_options", options);

    return mod;
}

/// Standalone build function for building rbitz directly (not as a dependency).
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Use helper to set up CRoaring - for standalone builds, use current directory
    const croaring = setupCRoaring(b, target, optimize, ".");

    // Install the static library if using local CRoaring
    if (croaring.lib) |lib| {
        b.installArtifact(lib);
    }

    // Create the rbitz module
    const mod = b.addModule("rbitz", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    linkCRoaring(mod, croaring);

    // Add version info
    const options = b.addOptions();
    options.addOption([]const u8, "version", getVersion(b));
    mod.addOptions("build_options", options);

    // Build executable
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
    linkCRoaring(exe.root_module, croaring);
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
    linkCRoaring(mod_tests.root_module, croaring);
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    linkCRoaring(exe_tests.root_module, croaring);
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
    linkCRoaring(bench.root_module, croaring);
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    run_bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bench.addArgs(args);
    }

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
}
