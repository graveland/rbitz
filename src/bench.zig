const std = @import("std");
const rbitz = @import("rbitz");

const Roaring32Bitmap = rbitz.Roaring32Bitmap;
const Roaring64Bitmap = rbitz.Roaring64Bitmap;

const Range = struct { first: u32, last: u32 };

// ============================================================================
// Configuration
// ============================================================================

const Config = struct {
    iterations: usize = 100,
    warmup: usize = 5,
    sizes: SizeSet = .{ .small = true, .medium = true, .large = true },
    format: Format = .text,
    filter: ?[]const u8 = null,

    const SizeSet = struct {
        small: bool = false,
        medium: bool = false,
        large: bool = false,
    };

    const Format = enum { text, json };
};

const Size = enum {
    small,
    medium,
    large,

    fn elementCount(self: Size) usize {
        return switch (self) {
            .small => 100,
            .medium => 10_000,
            .large => 1_000_000,
        };
    }

    fn name(self: Size) []const u8 {
        return switch (self) {
            .small => "S",
            .medium => "M",
            .large => "L",
        };
    }
};

// ============================================================================
// Memory Tracking Allocator
// ============================================================================

const TrackingAllocator = struct {
    backing: std.mem.Allocator,
    bytes_allocated: usize = 0,
    bytes_freed: usize = 0,
    allocation_count: usize = 0,
    free_count: usize = 0,
    peak_bytes: usize = 0,

    fn init(backing: std.mem.Allocator) TrackingAllocator {
        return .{ .backing = backing };
    }

    fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null; // Not supported
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawAlloc(len, alignment, ret_addr);
        if (result != null) {
            self.bytes_allocated += len;
            self.allocation_count += 1;
            const current = self.bytes_allocated -| self.bytes_freed;
            self.peak_bytes = @max(self.peak_bytes, current);
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.backing.rawResize(buf, alignment, new_len, ret_addr)) {
            if (new_len > buf.len) {
                self.bytes_allocated += new_len - buf.len;
            } else {
                self.bytes_freed += buf.len - new_len;
            }
            const current = self.bytes_allocated -| self.bytes_freed;
            self.peak_bytes = @max(self.peak_bytes, current);
            return true;
        }
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.backing.rawFree(buf, alignment, ret_addr);
        self.bytes_freed += buf.len;
        self.free_count += 1;
    }

    fn reset(self: *TrackingAllocator) void {
        self.bytes_allocated = 0;
        self.bytes_freed = 0;
        self.allocation_count = 0;
        self.free_count = 0;
        self.peak_bytes = 0;
    }

    fn currentBytes(self: *const TrackingAllocator) usize {
        return self.bytes_allocated -| self.bytes_freed;
    }
};

// ============================================================================
// Statistics
// ============================================================================

const Stats = struct {
    samples: std.ArrayList(u64),
    allocator: std.mem.Allocator,

    fn init(alloc: std.mem.Allocator) Stats {
        return .{
            .samples = .empty,
            .allocator = alloc,
        };
    }

    fn deinit(self: *Stats) void {
        self.samples.deinit(self.allocator);
    }

    fn add(self: *Stats, sample: u64) !void {
        try self.samples.append(self.allocator, sample);
    }

    fn min(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var result: u64 = std.math.maxInt(u64);
        for (self.samples.items) |s| {
            result = @min(result, s);
        }
        return result;
    }

    fn max(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var result: u64 = 0;
        for (self.samples.items) |s| {
            result = @max(result, s);
        }
        return result;
    }

    fn mean(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var sum: u128 = 0;
        for (self.samples.items) |s| {
            sum += s;
        }
        return @intCast(sum / self.samples.items.len);
    }

    fn median(self: *Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        std.mem.sort(u64, self.samples.items, {}, std.sort.asc(u64));
        const mid = self.samples.items.len / 2;
        if (self.samples.items.len % 2 == 0) {
            return (self.samples.items[mid - 1] + self.samples.items[mid]) / 2;
        }
        return self.samples.items[mid];
    }

    fn stddev(self: *const Stats) u64 {
        if (self.samples.items.len < 2) return 0;
        const avg = self.mean();
        var sum_sq: u128 = 0;
        for (self.samples.items) |s| {
            const diff: i128 = @as(i128, s) - @as(i128, avg);
            sum_sq += @intCast(@as(u128, @intCast(diff * diff)));
        }
        const variance = sum_sq / (self.samples.items.len - 1);
        return std.math.sqrt(variance);
    }
};

// ============================================================================
// Benchmark Result
// ============================================================================

const BenchmarkResult = struct {
    name: []const u8,
    size: Size,
    stats: struct {
        mean_ns: u64,
        std_ns: u64,
        min_ns: u64,
        max_ns: u64,
        median_ns: u64,
    },
    memory: struct {
        peak_bytes: usize,
        total_allocated: usize,
        allocation_count: usize,
    },
};

// ============================================================================
// Output Formatters
// ============================================================================

fn formatNanos(ns: u64) struct { value: f64, unit: []const u8 } {
    if (ns >= 1_000_000_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0, .unit = "s" };
    } else if (ns >= 1_000_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000_000.0, .unit = "ms" };
    } else if (ns >= 1_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000.0, .unit = "us" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(ns)), .unit = "ns" };
    }
}

fn formatBytes(bytes: usize) struct { value: f64, unit: []const u8 } {
    if (bytes >= 1024 * 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0), .unit = "GB" };
    } else if (bytes >= 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), .unit = "MB" };
    } else if (bytes >= 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / 1024.0, .unit = "KB" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(bytes)), .unit = "B" };
    }
}

fn formatOpsPerSec(ns: u64) struct { value: f64, unit: []const u8 } {
    if (ns == 0) return .{ .value = 0, .unit = "ops/s" };
    const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(ns));
    if (ops_per_sec >= 1_000_000_000) {
        return .{ .value = ops_per_sec / 1_000_000_000.0, .unit = "Gop/s" };
    } else if (ops_per_sec >= 1_000_000) {
        return .{ .value = ops_per_sec / 1_000_000.0, .unit = "Mop/s" };
    } else if (ops_per_sec >= 1_000) {
        return .{ .value = ops_per_sec / 1_000.0, .unit = "Kop/s" };
    } else {
        return .{ .value = ops_per_sec, .unit = "op/s" };
    }
}

fn printTextResult(result: BenchmarkResult, buf: []u8) void {
    const t = formatNanos(result.stats.mean_ns);
    const s = formatNanos(result.stats.std_ns);
    const mem = formatBytes(result.memory.peak_bytes);
    const ops = formatOpsPerSec(result.stats.mean_ns);

    const line = std.fmt.bufPrint(buf, "  {s:<28} {d:>7.3} {s:<2}  ±{d:>7.3} {s:<2}  {d:>7.2} {s:<5}  peak: {d:>6.1} {s:<2}  ({d} allocs)\n", .{
        result.name,
        t.value,
        t.unit,
        s.value,
        s.unit,
        ops.value,
        ops.unit,
        mem.value,
        mem.unit,
        result.memory.allocation_count,
    }) catch return;
    std.fs.File.stdout().writeAll(line) catch {};
}

fn printJsonResults(results: []const BenchmarkResult, config: Config, buf: []u8) void {
    const stdout = std.fs.File.stdout();

    stdout.writeAll("{\n") catch return;
    const cfg_line = std.fmt.bufPrint(buf, "  \"config\": {{\"iterations\": {d}, \"warmup\": {d}}},\n", .{ config.iterations, config.warmup }) catch return;
    stdout.writeAll(cfg_line) catch return;
    stdout.writeAll("  \"benchmarks\": [\n") catch return;

    for (results, 0..) |result, i| {
        stdout.writeAll("    {\n") catch return;
        const name_line = std.fmt.bufPrint(buf, "      \"name\": \"{s}\",\n", .{result.name}) catch return;
        stdout.writeAll(name_line) catch return;
        const size_line = std.fmt.bufPrint(buf, "      \"size\": \"{s}\",\n", .{result.size.name()}) catch return;
        stdout.writeAll(size_line) catch return;
        stdout.writeAll("      \"stats\": {\n") catch return;
        const mean_line = std.fmt.bufPrint(buf, "        \"mean_ns\": {d},\n", .{result.stats.mean_ns}) catch return;
        stdout.writeAll(mean_line) catch return;
        const std_line = std.fmt.bufPrint(buf, "        \"std_ns\": {d},\n", .{result.stats.std_ns}) catch return;
        stdout.writeAll(std_line) catch return;
        const min_line = std.fmt.bufPrint(buf, "        \"min_ns\": {d},\n", .{result.stats.min_ns}) catch return;
        stdout.writeAll(min_line) catch return;
        const max_line = std.fmt.bufPrint(buf, "        \"max_ns\": {d},\n", .{result.stats.max_ns}) catch return;
        stdout.writeAll(max_line) catch return;
        const median_line = std.fmt.bufPrint(buf, "        \"median_ns\": {d}\n", .{result.stats.median_ns}) catch return;
        stdout.writeAll(median_line) catch return;
        stdout.writeAll("      },\n") catch return;
        stdout.writeAll("      \"memory\": {\n") catch return;
        const peak_line = std.fmt.bufPrint(buf, "        \"peak_bytes\": {d},\n", .{result.memory.peak_bytes}) catch return;
        stdout.writeAll(peak_line) catch return;
        const total_line = std.fmt.bufPrint(buf, "        \"total_allocated\": {d},\n", .{result.memory.total_allocated}) catch return;
        stdout.writeAll(total_line) catch return;
        const alloc_line = std.fmt.bufPrint(buf, "        \"allocation_count\": {d}\n", .{result.memory.allocation_count}) catch return;
        stdout.writeAll(alloc_line) catch return;
        stdout.writeAll("      }\n") catch return;
        if (i < results.len - 1) {
            stdout.writeAll("    },\n") catch return;
        } else {
            stdout.writeAll("    }\n") catch return;
        }
    }

    stdout.writeAll("  ]\n") catch return;
    stdout.writeAll("}\n") catch return;
}

// ============================================================================
// Benchmark Framework
// ============================================================================

fn BenchmarkFn(comptime Context: type) type {
    return *const fn (*Context) void;
}

fn SetupFn(comptime Context: type) type {
    return *const fn (std.mem.Allocator, Size) anyerror!Context;
}

fn TeardownFn(comptime Context: type) type {
    return *const fn (*Context) void;
}

fn runBenchmark(
    comptime Context: type,
    name: []const u8,
    size: Size,
    config: Config,
    tracker: *TrackingAllocator,
    setup: SetupFn(Context),
    benchmark: BenchmarkFn(Context),
    teardown: TeardownFn(Context),
) !BenchmarkResult {
    var stats = Stats.init(tracker.backing);
    defer stats.deinit();

    var total_peak: usize = 0;
    var total_allocated: usize = 0;
    var total_allocs: usize = 0;

    // Warmup
    for (0..config.warmup) |_| {
        tracker.reset();
        var ctx = try setup(tracker.allocator(), size);
        benchmark(&ctx);
        teardown(&ctx);
    }

    // Timed runs
    for (0..config.iterations) |_| {
        tracker.reset();
        var ctx = try setup(tracker.allocator(), size);

        var timer = std.time.Timer.start() catch unreachable;
        benchmark(&ctx);
        const elapsed = timer.read();

        total_peak = @max(total_peak, tracker.peak_bytes);
        total_allocated += tracker.bytes_allocated;
        total_allocs += tracker.allocation_count;

        teardown(&ctx);
        try stats.add(elapsed);
    }

    return .{
        .name = name,
        .size = size,
        .stats = .{
            .mean_ns = stats.mean(),
            .std_ns = stats.stddev(),
            .min_ns = stats.min(),
            .max_ns = stats.max(),
            .median_ns = stats.median(),
        },
        .memory = .{
            .peak_bytes = total_peak,
            .total_allocated = total_allocated / config.iterations,
            .allocation_count = total_allocs / config.iterations,
        },
    };
}

// ============================================================================
// PRNG for reproducible random data
// ============================================================================

const Prng = struct {
    state: u64,

    fn init(seed: u64) Prng {
        return .{ .state = seed };
    }

    fn next(self: *Prng) u64 {
        var x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        return x;
    }

    fn nextBounded(self: *Prng, bound: u64) u64 {
        return self.next() % bound;
    }

    fn shuffle(self: *Prng, comptime T: type, items: []T) void {
        if (items.len < 2) return;
        var i = items.len - 1;
        while (i > 0) : (i -= 1) {
            const j = self.nextBounded(i + 1);
            const tmp = items[i];
            items[i] = items[@intCast(j)];
            items[@intCast(j)] = tmp;
        }
    }
};

// ============================================================================
// Core Operation Benchmarks (32-bit)
// ============================================================================

const AddSequentialCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    count: usize,

    fn setup(_: std.mem.Allocator, size: Size) !AddSequentialCtx {
        return .{
            .bitmap = Roaring32Bitmap.init(),
            .count = size.elementCount(),
        };
    }

    fn run(ctx: *AddSequentialCtx) void {
        const bm = ctx.bitmap orelse return;
        for (0..ctx.count) |i| {
            bm.add(@intCast(i));
        }
    }

    fn teardown(ctx: *AddSequentialCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const AddRandomCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    elements: []u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !AddRandomCtx {
        const count = size.elementCount();
        const elements = try alloc.alloc(u32, count);
        for (0..count) |i| {
            elements[i] = @intCast(i);
        }
        var prng = Prng.init(12345);
        prng.shuffle(u32, elements);

        return .{
            .bitmap = Roaring32Bitmap.init(),
            .elements = elements,
            .allocator = alloc,
        };
    }

    fn run(ctx: *AddRandomCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.elements) |elem| {
            bm.add(elem);
        }
    }

    fn teardown(ctx: *AddRandomCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.elements);
    }
};

const AddSparseCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    count: usize,

    fn setup(_: std.mem.Allocator, size: Size) !AddSparseCtx {
        return .{
            .bitmap = Roaring32Bitmap.init(),
            .count = size.elementCount(),
        };
    }

    fn run(ctx: *AddSparseCtx) void {
        const bm = ctx.bitmap orelse return;
        for (0..ctx.count) |i| {
            bm.add(@intCast(i * 10));
        }
    }

    fn teardown(ctx: *AddSparseCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const AddManyCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    elements: []u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !AddManyCtx {
        const count = size.elementCount();
        const elements = try alloc.alloc(u32, count);
        for (0..count) |i| {
            elements[i] = @intCast(i);
        }

        return .{
            .bitmap = Roaring32Bitmap.init(),
            .elements = elements,
            .allocator = alloc,
        };
    }

    fn run(ctx: *AddManyCtx) void {
        const bm = ctx.bitmap orelse return;
        bm.addMany(ctx.elements);
    }

    fn teardown(ctx: *AddManyCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.elements);
    }
};

const ContainsHitCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    queries: []u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !ContainsHitCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        for (0..count) |i| {
            bitmap.add(@intCast(i));
        }

        const queries = try alloc.alloc(u32, count);
        for (0..count) |i| {
            queries[i] = @intCast(i);
        }
        var prng = Prng.init(54321);
        prng.shuffle(u32, queries);

        return .{
            .bitmap = bitmap,
            .queries = queries,
            .allocator = alloc,
        };
    }

    fn run(ctx: *ContainsHitCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.queries) |q| {
            _ = bm.contains(q);
        }
    }

    fn teardown(ctx: *ContainsHitCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

const ContainsMissCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    queries: []u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !ContainsMissCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        // Insert sparse elements
        for (0..count) |i| {
            bitmap.add(@intCast(i * 10));
        }

        // Query elements that don't exist
        const queries = try alloc.alloc(u32, count);
        for (0..count) |i| {
            queries[i] = @as(u32, @intCast(i * 10)) + 5;
        }
        var prng = Prng.init(67890);
        prng.shuffle(u32, queries);

        return .{
            .bitmap = bitmap,
            .queries = queries,
            .allocator = alloc,
        };
    }

    fn run(ctx: *ContainsMissCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.queries) |q| {
            _ = bm.contains(q);
        }
    }

    fn teardown(ctx: *ContainsMissCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

const RemoveSequentialCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    count: usize,

    fn setup(_: std.mem.Allocator, size: Size) !RemoveSequentialCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        for (0..count) |i| {
            bitmap.add(@intCast(i));
        }
        return .{ .bitmap = bitmap, .count = count };
    }

    fn run(ctx: *RemoveSequentialCtx) void {
        const bm = ctx.bitmap orelse return;
        for (0..ctx.count) |i| {
            bm.remove(@intCast(i));
        }
    }

    fn teardown(ctx: *RemoveSequentialCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const RemoveRandomCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    elements: []u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !RemoveRandomCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const elements = try alloc.alloc(u32, count);

        for (0..count) |i| {
            bitmap.add(@intCast(i));
            elements[i] = @intCast(i);
        }
        var prng = Prng.init(11111);
        prng.shuffle(u32, elements);

        return .{
            .bitmap = bitmap,
            .elements = elements,
            .allocator = alloc,
        };
    }

    fn run(ctx: *RemoveRandomCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.elements) |elem| {
            bm.remove(elem);
        }
    }

    fn teardown(ctx: *RemoveRandomCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.elements);
    }
};

// ============================================================================
// Range Operation Benchmarks
// ============================================================================

const AddRangeCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    count: usize,

    fn setup(_: std.mem.Allocator, size: Size) !AddRangeCtx {
        return .{
            .bitmap = Roaring32Bitmap.init(),
            .count = size.elementCount() / 10,
        };
    }

    fn run(ctx: *AddRangeCtx) void {
        const bm = ctx.bitmap orelse return;
        for (0..ctx.count) |i| {
            const start: u32 = @intCast(i * 20);
            bm.addRangeClosed(start, start + 9);
        }
    }

    fn teardown(ctx: *AddRangeCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const AddRangeLargeCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    count: usize,

    fn setup(_: std.mem.Allocator, size: Size) !AddRangeLargeCtx {
        return .{
            .bitmap = Roaring32Bitmap.init(),
            .count = @max(1, size.elementCount() / 1000),
        };
    }

    fn run(ctx: *AddRangeLargeCtx) void {
        const bm = ctx.bitmap orelse return;
        for (0..ctx.count) |i| {
            const start: u32 = @intCast(i * 2000);
            bm.addRangeClosed(start, start + 999);
        }
    }

    fn teardown(ctx: *AddRangeLargeCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const ContainsRangeHitCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !ContainsRangeHitCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        // Insert contiguous elements
        bitmap.addRange(0, count);

        const num_queries = @max(1, count / 100);
        const ranges = try alloc.alloc(Range, num_queries);
        var prng = Prng.init(22222);
        for (0..num_queries) |i| {
            const start: u32 = @intCast(prng.nextBounded(count - 10));
            ranges[i] = .{ .first = start, .last = start + 9 };
        }

        return .{
            .bitmap = bitmap,
            .ranges = ranges,
            .allocator = alloc,
        };
    }

    fn run(ctx: *ContainsRangeHitCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.ranges) |r| {
            _ = bm.containsRange(r.first, r.last + 1); // containsRange uses [start, end)
        }
    }

    fn teardown(ctx: *ContainsRangeHitCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

const RemoveRangeCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !RemoveRangeCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap.addRange(0, count);

        const num_ranges = @max(1, count / 100);
        const ranges = try alloc.alloc(Range, num_ranges);
        for (0..num_ranges) |i| {
            const start: u32 = @intCast(i * 100);
            ranges[i] = .{ .first = start, .last = start + 9 };
        }

        return .{
            .bitmap = bitmap,
            .ranges = ranges,
            .allocator = alloc,
        };
    }

    fn run(ctx: *RemoveRangeCtx) void {
        const bm = ctx.bitmap orelse return;
        for (ctx.ranges) |r| {
            bm.removeRangeClosed(r.first, r.last);
        }
    }

    fn teardown(ctx: *RemoveRangeCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

// ============================================================================
// Set Operation Benchmarks
// ============================================================================

const UnionCtx = struct {
    bitmap1: ?*Roaring32Bitmap,
    bitmap2: ?*Roaring32Bitmap,
    result: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !UnionCtx {
        const count = size.elementCount();
        const bitmap1 = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const bitmap2 = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        // Disjoint ranges
        bitmap1.addRange(0, count);
        bitmap2.addRange(count * 2, count * 3);

        return .{
            .bitmap1 = bitmap1,
            .bitmap2 = bitmap2,
            .result = null,
        };
    }

    fn run(ctx: *UnionCtx) void {
        const bm1 = ctx.bitmap1 orelse return;
        const bm2 = ctx.bitmap2 orelse return;
        ctx.result = bm1.unionWith(bm2);
    }

    fn teardown(ctx: *UnionCtx) void {
        if (ctx.result) |r| r.deinit();
        if (ctx.bitmap1) |bm| bm.deinit();
        if (ctx.bitmap2) |bm| bm.deinit();
    }
};

const UnionInplaceCtx = struct {
    bitmap1: ?*Roaring32Bitmap,
    bitmap2: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !UnionInplaceCtx {
        const count = size.elementCount();
        const bitmap1 = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const bitmap2 = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap1.addRange(0, count);
        bitmap2.addRange(count / 2, count + count / 2);

        return .{
            .bitmap1 = bitmap1,
            .bitmap2 = bitmap2,
        };
    }

    fn run(ctx: *UnionInplaceCtx) void {
        const bm1 = ctx.bitmap1 orelse return;
        const bm2 = ctx.bitmap2 orelse return;
        bm1.unionInplace(bm2);
    }

    fn teardown(ctx: *UnionInplaceCtx) void {
        if (ctx.bitmap1) |bm| bm.deinit();
        if (ctx.bitmap2) |bm| bm.deinit();
    }
};

const IntersectionCtx = struct {
    bitmap1: ?*Roaring32Bitmap,
    bitmap2: ?*Roaring32Bitmap,
    result: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !IntersectionCtx {
        const count = size.elementCount();
        const bitmap1 = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const bitmap2 = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap1.addRange(0, count);
        bitmap2.addRange(count / 2, count + count / 2);

        return .{
            .bitmap1 = bitmap1,
            .bitmap2 = bitmap2,
            .result = null,
        };
    }

    fn run(ctx: *IntersectionCtx) void {
        const bm1 = ctx.bitmap1 orelse return;
        const bm2 = ctx.bitmap2 orelse return;
        ctx.result = bm1.intersectionWith(bm2);
    }

    fn teardown(ctx: *IntersectionCtx) void {
        if (ctx.result) |r| r.deinit();
        if (ctx.bitmap1) |bm| bm.deinit();
        if (ctx.bitmap2) |bm| bm.deinit();
    }
};

const DifferenceCtx = struct {
    bitmap1: ?*Roaring32Bitmap,
    bitmap2: ?*Roaring32Bitmap,
    result: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !DifferenceCtx {
        const count = size.elementCount();
        const bitmap1 = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const bitmap2 = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap1.addRange(0, count);
        bitmap2.addRange(count / 2, count + count / 2);

        return .{
            .bitmap1 = bitmap1,
            .bitmap2 = bitmap2,
            .result = null,
        };
    }

    fn run(ctx: *DifferenceCtx) void {
        const bm1 = ctx.bitmap1 orelse return;
        const bm2 = ctx.bitmap2 orelse return;
        ctx.result = bm1.differenceWith(bm2);
    }

    fn teardown(ctx: *DifferenceCtx) void {
        if (ctx.result) |r| r.deinit();
        if (ctx.bitmap1) |bm| bm.deinit();
        if (ctx.bitmap2) |bm| bm.deinit();
    }
};

const XorCtx = struct {
    bitmap1: ?*Roaring32Bitmap,
    bitmap2: ?*Roaring32Bitmap,
    result: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !XorCtx {
        const count = size.elementCount();
        const bitmap1 = Roaring32Bitmap.init() orelse return error.OutOfMemory;
        const bitmap2 = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap1.addRange(0, count);
        bitmap2.addRange(count / 2, count + count / 2);

        return .{
            .bitmap1 = bitmap1,
            .bitmap2 = bitmap2,
            .result = null,
        };
    }

    fn run(ctx: *XorCtx) void {
        const bm1 = ctx.bitmap1 orelse return;
        const bm2 = ctx.bitmap2 orelse return;
        ctx.result = bm1.xorWith(bm2);
    }

    fn teardown(ctx: *XorCtx) void {
        if (ctx.result) |r| r.deinit();
        if (ctx.bitmap1) |bm| bm.deinit();
        if (ctx.bitmap2) |bm| bm.deinit();
    }
};

// ============================================================================
// Traversal Benchmarks
// ============================================================================

const IteratorCtx = struct {
    bitmap: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !IteratorCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        // Create sparse data
        for (0..count) |i| {
            bitmap.add(@intCast(i * 3));
        }

        return .{ .bitmap = bitmap };
    }

    fn run(ctx: *IteratorCtx) void {
        const bm = ctx.bitmap orelse return;
        var it = Roaring32Bitmap.Iterator.init(bm);
        var sum: u64 = 0;
        while (it.next()) |val| {
            sum += val;
        }
        std.mem.doNotOptimizeAway(sum);
    }

    fn teardown(ctx: *IteratorCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const ToArrayCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    result: ?[]u32,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !ToArrayCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        for (0..count) |i| {
            bitmap.add(@intCast(i * 3));
        }

        return .{
            .bitmap = bitmap,
            .result = null,
            .allocator = alloc,
        };
    }

    fn run(ctx: *ToArrayCtx) void {
        const bm = ctx.bitmap orelse return;
        ctx.result = bm.toArray(ctx.allocator) catch null;
    }

    fn teardown(ctx: *ToArrayCtx) void {
        if (ctx.result) |r| ctx.allocator.free(r);
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const CardinalityCtx = struct {
    bitmap: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !CardinalityCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap.addRange(0, count);

        return .{ .bitmap = bitmap };
    }

    fn run(ctx: *CardinalityCtx) void {
        const bm = ctx.bitmap orelse return;
        const c = bm.getCardinality();
        std.mem.doNotOptimizeAway(c);
    }

    fn teardown(ctx: *CardinalityCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

// ============================================================================
// Serialization Benchmarks
// ============================================================================

const SerializeCtx = struct {
    bitmap: ?*Roaring32Bitmap,
    buffer: []u8,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !SerializeCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap.addRange(0, count);
        _ = bitmap.runOptimize();

        const buf_size = bitmap.portableSizeInBytes();
        const buffer = try alloc.alloc(u8, buf_size);

        return .{
            .bitmap = bitmap,
            .buffer = buffer,
            .allocator = alloc,
        };
    }

    fn run(ctx: *SerializeCtx) void {
        const bm = ctx.bitmap orelse return;
        _ = bm.portableSerialize(ctx.buffer);
    }

    fn teardown(ctx: *SerializeCtx) void {
        ctx.allocator.free(ctx.buffer);
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

const DeserializeCtx = struct {
    buffer: []u8,
    result: ?*Roaring32Bitmap,
    allocator: std.mem.Allocator,

    fn setup(alloc: std.mem.Allocator, size: Size) !DeserializeCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        bitmap.addRange(0, count);
        _ = bitmap.runOptimize();

        const buf_size = bitmap.portableSizeInBytes();
        const buffer = try alloc.alloc(u8, buf_size);
        _ = bitmap.portableSerialize(buffer);
        bitmap.deinit();

        return .{
            .buffer = buffer,
            .result = null,
            .allocator = alloc,
        };
    }

    fn run(ctx: *DeserializeCtx) void {
        ctx.result = Roaring32Bitmap.portableDeserialize(ctx.buffer);
    }

    fn teardown(ctx: *DeserializeCtx) void {
        if (ctx.result) |r| r.deinit();
        ctx.allocator.free(ctx.buffer);
    }
};

// ============================================================================
// Optimization Benchmarks
// ============================================================================

const RunOptimizeCtx = struct {
    bitmap: ?*Roaring32Bitmap,

    fn setup(_: std.mem.Allocator, size: Size) !RunOptimizeCtx {
        const count = size.elementCount();
        const bitmap = Roaring32Bitmap.init() orelse return error.OutOfMemory;

        // Add sequential data (good for run optimization)
        bitmap.addRange(0, count);

        return .{ .bitmap = bitmap };
    }

    fn run(ctx: *RunOptimizeCtx) void {
        const bm = ctx.bitmap orelse return;
        _ = bm.runOptimize();
    }

    fn teardown(ctx: *RunOptimizeCtx) void {
        if (ctx.bitmap) |bm| bm.deinit();
    }
};

// ============================================================================
// CLI Argument Parsing
// ============================================================================

fn parseArgs(alloc: std.mem.Allocator) !Config {
    var config = Config{};

    var args = try std.process.argsWithAllocator(alloc);
    defer args.deinit();

    _ = args.skip();

    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--filter=")) {
            config.filter = arg["--filter=".len..];
        } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
            config.iterations = std.fmt.parseInt(usize, arg["--iterations=".len..], 10) catch 100;
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            config.warmup = std.fmt.parseInt(usize, arg["--warmup=".len..], 10) catch 5;
        } else if (std.mem.startsWith(u8, arg, "--size=")) {
            const sizes_str = arg["--size=".len..];
            config.sizes = .{ .small = false, .medium = false, .large = false };
            var it = std.mem.splitScalar(u8, sizes_str, ',');
            while (it.next()) |s| {
                if (std.mem.eql(u8, s, "S")) config.sizes.small = true;
                if (std.mem.eql(u8, s, "M")) config.sizes.medium = true;
                if (std.mem.eql(u8, s, "L")) config.sizes.large = true;
            }
        } else if (std.mem.eql(u8, arg, "--format=json")) {
            config.format = .json;
        } else if (std.mem.eql(u8, arg, "--format=text")) {
            config.format = .text;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }

    return config;
}

fn printHelp() void {
    const help =
        \\RBITZ (Roaring Bitmap) Benchmarks
        \\
        \\Usage: bench [options]
        \\
        \\Options:
        \\  --filter=<pattern>    Run only benchmarks containing pattern
        \\  --iterations=<N>      Number of timed iterations (default: 100)
        \\  --warmup=<N>          Number of warmup iterations (default: 5)
        \\  --size=<S,M,L>        Comma-separated sizes to test (default: S,M,L)
        \\  --format=<text|json>  Output format (default: text)
        \\  --help, -h            Show this help message
        \\
        \\Examples:
        \\  bench                          Run all benchmarks
        \\  bench --filter=add             Run only add benchmarks
        \\  bench --size=M --iterations=50 Run medium size with 50 iterations
        \\  bench --format=json            Output JSON for tooling
        \\
    ;
    std.fs.File.stdout().writeAll(help) catch {};
}

fn matchesFilter(name: []const u8, filter: ?[]const u8) bool {
    if (filter) |f| {
        return std.mem.indexOf(u8, name, f) != null;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

fn print(buf: []u8, comptime fmt: []const u8, args: anytype) void {
    const line = std.fmt.bufPrint(buf, fmt, args) catch return;
    std.fs.File.stdout().writeAll(line) catch {};
}

fn write(str: []const u8) void {
    std.fs.File.stdout().writeAll(str) catch {};
}

pub fn main() !void {
    var gpa_state: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const config = try parseArgs(gpa);

    var results: std.ArrayList(BenchmarkResult) = .empty;
    defer results.deinit(gpa);

    var tracker = TrackingAllocator.init(gpa);
    var buf: [4096]u8 = undefined;

    if (config.format == .text) {
        write("RBITZ (Roaring Bitmap) Benchmarks\n");
        write("==================================\n");
        print(&buf, "Config: iterations={d}, warmup={d}\n\n", .{ config.iterations, config.warmup });
    }

    const sizes = [_]Size{ .small, .medium, .large };
    const size_enabled = [_]bool{ config.sizes.small, config.sizes.medium, config.sizes.large };

    for (sizes, size_enabled) |size, enabled| {
        if (!enabled) continue;

        if (config.format == .text) {
            print(&buf, "Size: {s} (N={d})\n", .{ size.name(), size.elementCount() });
            write("------------------------------------------------------------\n");
        }

        // Core Operations
        if (matchesFilter("add/sequential", config.filter)) {
            const r = try runBenchmark(AddSequentialCtx, "add/sequential", size, config, &tracker, AddSequentialCtx.setup, AddSequentialCtx.run, AddSequentialCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("add/random", config.filter)) {
            const r = try runBenchmark(AddRandomCtx, "add/random", size, config, &tracker, AddRandomCtx.setup, AddRandomCtx.run, AddRandomCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("add/sparse", config.filter)) {
            const r = try runBenchmark(AddSparseCtx, "add/sparse", size, config, &tracker, AddSparseCtx.setup, AddSparseCtx.run, AddSparseCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("addMany", config.filter)) {
            const r = try runBenchmark(AddManyCtx, "addMany", size, config, &tracker, AddManyCtx.setup, AddManyCtx.run, AddManyCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("contains/hit", config.filter)) {
            const r = try runBenchmark(ContainsHitCtx, "contains/hit", size, config, &tracker, ContainsHitCtx.setup, ContainsHitCtx.run, ContainsHitCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("contains/miss", config.filter)) {
            const r = try runBenchmark(ContainsMissCtx, "contains/miss", size, config, &tracker, ContainsMissCtx.setup, ContainsMissCtx.run, ContainsMissCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("remove/sequential", config.filter)) {
            const r = try runBenchmark(RemoveSequentialCtx, "remove/sequential", size, config, &tracker, RemoveSequentialCtx.setup, RemoveSequentialCtx.run, RemoveSequentialCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("remove/random", config.filter)) {
            const r = try runBenchmark(RemoveRandomCtx, "remove/random", size, config, &tracker, RemoveRandomCtx.setup, RemoveRandomCtx.run, RemoveRandomCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Range Operations
        if (matchesFilter("addRange/small", config.filter)) {
            const r = try runBenchmark(AddRangeCtx, "addRange/small", size, config, &tracker, AddRangeCtx.setup, AddRangeCtx.run, AddRangeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("addRange/large", config.filter)) {
            const r = try runBenchmark(AddRangeLargeCtx, "addRange/large", size, config, &tracker, AddRangeLargeCtx.setup, AddRangeLargeCtx.run, AddRangeLargeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("containsRange", config.filter)) {
            const r = try runBenchmark(ContainsRangeHitCtx, "containsRange", size, config, &tracker, ContainsRangeHitCtx.setup, ContainsRangeHitCtx.run, ContainsRangeHitCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("removeRange", config.filter)) {
            const r = try runBenchmark(RemoveRangeCtx, "removeRange", size, config, &tracker, RemoveRangeCtx.setup, RemoveRangeCtx.run, RemoveRangeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Set Operations
        if (matchesFilter("union", config.filter)) {
            const r = try runBenchmark(UnionCtx, "union", size, config, &tracker, UnionCtx.setup, UnionCtx.run, UnionCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("unionInplace", config.filter)) {
            const r = try runBenchmark(UnionInplaceCtx, "unionInplace", size, config, &tracker, UnionInplaceCtx.setup, UnionInplaceCtx.run, UnionInplaceCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("intersection", config.filter)) {
            const r = try runBenchmark(IntersectionCtx, "intersection", size, config, &tracker, IntersectionCtx.setup, IntersectionCtx.run, IntersectionCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("difference", config.filter)) {
            const r = try runBenchmark(DifferenceCtx, "difference", size, config, &tracker, DifferenceCtx.setup, DifferenceCtx.run, DifferenceCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("xor", config.filter)) {
            const r = try runBenchmark(XorCtx, "xor", size, config, &tracker, XorCtx.setup, XorCtx.run, XorCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Traversal
        if (matchesFilter("iterator", config.filter)) {
            const r = try runBenchmark(IteratorCtx, "iterator", size, config, &tracker, IteratorCtx.setup, IteratorCtx.run, IteratorCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("toArray", config.filter)) {
            const r = try runBenchmark(ToArrayCtx, "toArray", size, config, &tracker, ToArrayCtx.setup, ToArrayCtx.run, ToArrayCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("cardinality", config.filter)) {
            const r = try runBenchmark(CardinalityCtx, "cardinality", size, config, &tracker, CardinalityCtx.setup, CardinalityCtx.run, CardinalityCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Serialization
        if (matchesFilter("serialize", config.filter)) {
            const r = try runBenchmark(SerializeCtx, "serialize", size, config, &tracker, SerializeCtx.setup, SerializeCtx.run, SerializeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("deserialize", config.filter)) {
            const r = try runBenchmark(DeserializeCtx, "deserialize", size, config, &tracker, DeserializeCtx.setup, DeserializeCtx.run, DeserializeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Optimization
        if (matchesFilter("runOptimize", config.filter)) {
            const r = try runBenchmark(RunOptimizeCtx, "runOptimize", size, config, &tracker, RunOptimizeCtx.setup, RunOptimizeCtx.run, RunOptimizeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (config.format == .text) {
            write("\n");
        }
    }

    if (config.format == .json) {
        printJsonResults(results.items, config, &buf);
    }
}
