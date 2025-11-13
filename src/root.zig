//! Zig wrapper for CRoaring - Roaring Bitmaps in C
//!
//! Roaring bitmaps are compressed bitmaps which tend to outperform conventional
//! compressed bitmaps such as WAH, EWAH or Concise.
//!
//! This implementation uses manual FFI bindings instead of @cImport to avoid
//! ARM NEON header translation issues on Apple Silicon.

const std = @import("std");

/// Opaque type for roaring_bitmap_t
pub const roaring_bitmap_t = opaque {};

/// Container type (opaque in the API)
pub const container_t = opaque {};

/// Bulk operation context
pub const roaring_bulk_context_t = extern struct {
    container: ?*container_t,
    idx: c_int,
    key: u16,
    typecode: u8,
};

/// Iterator for roaring bitmaps
pub const roaring_uint32_iterator_t = extern struct {
    parent: *const roaring_bitmap_t,
    container: ?*const container_t,
    typecode: u8,
    container_index: i32,
    highbits: u32,
    container_it: extern struct {
        index: i32,
    },
    current_value: u32,
    has_value: bool,
};

// External C functions from CRoaring
extern "c" fn roaring_bitmap_create() ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_create_with_capacity(cap: u32) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_of_ptr(n_args: usize, vals: [*]const u32) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_from_range(min: u64, max: u64, step: u32) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_free(r: *const roaring_bitmap_t) void;
extern "c" fn roaring_bitmap_copy(r: *const roaring_bitmap_t) ?*roaring_bitmap_t;

extern "c" fn roaring_bitmap_add(r: *roaring_bitmap_t, x: u32) void;
extern "c" fn roaring_bitmap_add_checked(r: *roaring_bitmap_t, x: u32) bool;
extern "c" fn roaring_bitmap_add_many(r: *roaring_bitmap_t, n_args: usize, vals: [*]const u32) void;
extern "c" fn roaring_bitmap_add_range(r: *roaring_bitmap_t, min: u64, max: u64) void;
extern "c" fn roaring_bitmap_add_range_closed(r: *roaring_bitmap_t, min: u32, max: u32) void;

extern "c" fn roaring_bitmap_remove(r: *roaring_bitmap_t, x: u32) void;
extern "c" fn roaring_bitmap_remove_checked(r: *roaring_bitmap_t, x: u32) bool;
extern "c" fn roaring_bitmap_remove_range(r: *roaring_bitmap_t, min: u64, max: u64) void;
extern "c" fn roaring_bitmap_remove_range_closed(r: *roaring_bitmap_t, min: u32, max: u32) void;
extern "c" fn roaring_bitmap_remove_many(r: *roaring_bitmap_t, n_args: usize, vals: [*]const u32) void;

extern "c" fn roaring_bitmap_contains(r: *const roaring_bitmap_t, val: u32) bool;
extern "c" fn roaring_bitmap_contains_range(r: *const roaring_bitmap_t, range_start: u64, range_end: u64) bool;
extern "c" fn roaring_bitmap_get_cardinality(r: *const roaring_bitmap_t) u64;
extern "c" fn roaring_bitmap_is_empty(r: *const roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_clear(r: *roaring_bitmap_t) void;
extern "c" fn roaring_bitmap_minimum(r: *const roaring_bitmap_t) u32;
extern "c" fn roaring_bitmap_maximum(r: *const roaring_bitmap_t) u32;

extern "c" fn roaring_bitmap_or(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_or_inplace(r1: *roaring_bitmap_t, r2: *const roaring_bitmap_t) void;
extern "c" fn roaring_bitmap_and(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_and_inplace(r1: *roaring_bitmap_t, r2: *const roaring_bitmap_t) void;
extern "c" fn roaring_bitmap_xor(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_xor_inplace(r1: *roaring_bitmap_t, r2: *const roaring_bitmap_t) void;
extern "c" fn roaring_bitmap_andnot(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_andnot_inplace(r1: *roaring_bitmap_t, r2: *const roaring_bitmap_t) void;

extern "c" fn roaring_bitmap_intersect(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_equals(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_is_subset(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_is_strict_subset(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_jaccard_index(r1: *const roaring_bitmap_t, r2: *const roaring_bitmap_t) f64;

extern "c" fn roaring_bitmap_flip(r1: *const roaring_bitmap_t, range_start: u64, range_end: u64) ?*roaring_bitmap_t;
extern "c" fn roaring_bitmap_flip_inplace(r1: *roaring_bitmap_t, range_start: u64, range_end: u64) void;

extern "c" fn roaring_bitmap_rank(r: *const roaring_bitmap_t, x: u32) u64;
extern "c" fn roaring_bitmap_select(r: *const roaring_bitmap_t, rank: u32, element: *u32) bool;
extern "c" fn roaring_bitmap_get_index(r: *const roaring_bitmap_t, x: u32) i64;

extern "c" fn roaring_bitmap_to_uint32_array(r: *const roaring_bitmap_t, ans: [*]u32) void;
extern "c" fn roaring_bitmap_run_optimize(r: *roaring_bitmap_t) bool;
extern "c" fn roaring_bitmap_shrink_to_fit(r: *roaring_bitmap_t) usize;

extern "c" fn roaring_bitmap_portable_size_in_bytes(r: *const roaring_bitmap_t) usize;
extern "c" fn roaring_bitmap_portable_serialize(r: *const roaring_bitmap_t, buf: [*]u8) usize;
extern "c" fn roaring_bitmap_portable_deserialize_safe(buf: [*]const u8, maxbytes: usize) ?*roaring_bitmap_t;

extern "c" fn roaring_bitmap_printf(r: *const roaring_bitmap_t) void;

extern "c" fn roaring_iterator_init(r: *const roaring_bitmap_t, newit: *roaring_uint32_iterator_t) void;
extern "c" fn roaring_uint32_iterator_advance(it: *roaring_uint32_iterator_t) bool;

/// Roaring32Bitmap is a compressed bitmap data structure for 32-bit integers.
pub const Roaring32Bitmap = struct {
    ptr: *roaring_bitmap_t,

    // init creates a new roaring bitmap
    pub fn init() ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_create() orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring32Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    /// initWithCapacity creates a new roaring bitmap with a given capacity hint
    pub fn initWithCapacity(capacity: u32) ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_create_with_capacity(capacity) orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring32Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    /// initFromSlice creates a new bitmap from a slice of uint32 integers
    pub fn initFromSlice(values: []const u32) ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_of_ptr(values.len, values.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring32Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    /// initFromRange creates a bitmap from a range [min, max) with step
    pub fn initFromRange(min: u64, max: u64, step: u32) ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_from_range(min, max, step) orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring32Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    pub fn deinit(self: *Roaring32Bitmap) void {
        roaring_bitmap_free(self.ptr);
        const allocator = std.heap.c_allocator;
        allocator.destroy(self);
    }

    /// copy a bitmap (performs memory allocation)
    pub fn copy(self: *const Roaring32Bitmap) ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_copy(self.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const new_self = allocator.create(Roaring32Bitmap) catch return null;
        new_self.* = .{ .ptr = bitmap_ptr };
        return new_self;
    }

    /// add a value to the bitmap
    pub fn add(self: *Roaring32Bitmap, value: u32) void {
        roaring_bitmap_add(self.ptr, value);
    }

    /// addChecked adds a value to the bitmap and returns true if the value was added (wasn't present before)
    pub fn addChecked(self: *Roaring32Bitmap, value: u32) bool {
        return roaring_bitmap_add_checked(self.ptr, value);
    }

    /// addMany adds multiple values efficiently
    pub fn addMany(self: *Roaring32Bitmap, values: []const u32) void {
        roaring_bitmap_add_many(self.ptr, values.len, values.ptr);
    }

    /// addRange adds all values in the range [min, max)
    pub fn addRange(self: *Roaring32Bitmap, min: u64, max: u64) void {
        roaring_bitmap_add_range(self.ptr, min, max);
    }

    /// addRangeClosed adds all values in the range [min, max] (inclusive)
    pub fn addRangeClosed(self: *Roaring32Bitmap, min: u32, max: u32) void {
        roaring_bitmap_add_range_closed(self.ptr, min, max);
    }

    /// remove a value from the bitmap
    pub fn remove(self: *Roaring32Bitmap, value: u32) void {
        roaring_bitmap_remove(self.ptr, value);
    }

    /// removeChecked removes a value from the bitmap and returns true if the value was removed (was present before)
    pub fn removeChecked(self: *Roaring32Bitmap, value: u32) bool {
        return roaring_bitmap_remove_checked(self.ptr, value);
    }

    /// removeRange removes all values in the range [min, max)
    pub fn removeRange(self: *Roaring32Bitmap, min: u64, max: u64) void {
        roaring_bitmap_remove_range(self.ptr, min, max);
    }

    /// removeRangeClosed remove all values in the range [min, max] (inclusive)
    pub fn removeRangeClosed(self: *Roaring32Bitmap, min: u32, max: u32) void {
        roaring_bitmap_remove_range_closed(self.ptr, min, max);
    }

    /// removeMany removes multiple values
    pub fn removeMany(self: *Roaring32Bitmap, values: []const u32) void {
        roaring_bitmap_remove_many(self.ptr, values.len, values.ptr);
    }

    /// contains checks if a value is present in the bitmap
    pub fn contains(self: *const Roaring32Bitmap, value: u32) bool {
        return roaring_bitmap_contains(self.ptr, value);
    }

    /// containsRange checks if a range [start, end) is present in the bitmap
    pub fn containsRange(self: *const Roaring32Bitmap, start: u64, end: u64) bool {
        return roaring_bitmap_contains_range(self.ptr, start, end);
    }

    /// getCardinality returns the number of elements in the bitmap
    pub fn getCardinality(self: *const Roaring32Bitmap) u64 {
        return roaring_bitmap_get_cardinality(self.ptr);
    }

    /// isEmpty returns true if the bitmap is empty
    pub fn isEmpty(self: *const Roaring32Bitmap) bool {
        return roaring_bitmap_is_empty(self.ptr);
    }

    /// clear the bitmap (remove all elements)
    pub fn clear(self: *Roaring32Bitmap) void {
        roaring_bitmap_clear(self.ptr);
    }

    /// minimum gets the minimum value in the bitmap, or null if empty
    pub fn minimum(self: *const Roaring32Bitmap) ?u32 {
        if (self.isEmpty()) return null;
        return roaring_bitmap_minimum(self.ptr);
    }

    /// maximum gets the maximum value in the bitmap, or null if empty.
    pub fn maximum(self: *const Roaring32Bitmap) ?u32 {
        if (self.isEmpty()) return null;
        return roaring_bitmap_maximum(self.ptr);
    }

    /// unionWith computes the union of two bitmaps
    pub fn unionWith(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) ?*Roaring32Bitmap {
        const result_ptr = roaring_bitmap_or(self.ptr, other.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const result = allocator.create(Roaring32Bitmap) catch return null;
        result.* = .{ .ptr = result_ptr };
        return result;
    }

    /// unionInplace computes the union in place (modifies self)
    pub fn unionInplace(self: *Roaring32Bitmap, other: *const Roaring32Bitmap) void {
        roaring_bitmap_or_inplace(self.ptr, other.ptr);
    }

    /// intersectionWith computes the intersection of two bitmaps
    pub fn intersectionWith(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) ?*Roaring32Bitmap {
        const result_ptr = roaring_bitmap_and(self.ptr, other.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const result = allocator.create(Roaring32Bitmap) catch return null;
        result.* = .{ .ptr = result_ptr };
        return result;
    }

    /// intersectionInplace computes the intersection in place (modifies self)
    pub fn intersectionInplace(self: *Roaring32Bitmap, other: *const Roaring32Bitmap) void {
        roaring_bitmap_and_inplace(self.ptr, other.ptr);
    }

    /// intersects checks if two bitmaps intersect (have any common elements)
    pub fn intersects(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) bool {
        return roaring_bitmap_intersect(self.ptr, other.ptr);
    }

    /// xorWIth computes the XOR (symmetric difference) of two bitmaps
    pub fn xorWith(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) ?*Roaring32Bitmap {
        const result_ptr = roaring_bitmap_xor(self.ptr, other.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const result = allocator.create(Roaring32Bitmap) catch return null;
        result.* = .{ .ptr = result_ptr };
        return result;
    }

    /// xorInplace computes the XOR in place (modifies self)
    pub fn xorInplace(self: *Roaring32Bitmap, other: *const Roaring32Bitmap) void {
        roaring_bitmap_xor_inplace(self.ptr, other.ptr);
    }

    /// differenceWith computes the difference (andnot) between two bitmaps
    pub fn differenceWith(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) ?*Roaring32Bitmap {
        const result_ptr = roaring_bitmap_andnot(self.ptr, other.ptr) orelse return null;
        const allocator = std.heap.c_allocator;
        const result = allocator.create(Roaring32Bitmap) catch return null;
        result.* = .{ .ptr = result_ptr };
        return result;
    }

    /// differenceInplace computes the difference in place (modifies self)
    pub fn differenceInplace(self: *Roaring32Bitmap, other: *const Roaring32Bitmap) void {
        roaring_bitmap_andnot_inplace(self.ptr, other.ptr);
    }

    /// equals checks if two bitmaps are equal
    pub fn equals(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) bool {
        return roaring_bitmap_equals(self.ptr, other.ptr);
    }

    /// isSubset checks if self is a subset of other
    pub fn isSubset(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) bool {
        return roaring_bitmap_is_subset(self.ptr, other.ptr);
    }

    /// isStrictSubset checks if self is a strict subset of other
    pub fn isStrictSubset(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) bool {
        return roaring_bitmap_is_strict_subset(self.ptr, other.ptr);
    }

    /// jaccardIndex computes the Jaccard index between two bitmaps
    pub fn jaccardIndex(self: *const Roaring32Bitmap, other: *const Roaring32Bitmap) f64 {
        return roaring_bitmap_jaccard_index(self.ptr, other.ptr);
    }

    /// flip (negate) bits in the range [start, end)
    pub fn flip(self: *const Roaring32Bitmap, start: u64, end: u64) ?*Roaring32Bitmap {
        const result_ptr = roaring_bitmap_flip(self.ptr, start, end) orelse return null;
        const allocator = std.heap.c_allocator;
        const result = allocator.create(Roaring32Bitmap) catch return null;
        result.* = .{ .ptr = result_ptr };
        return result;
    }

    /// flipInplace (negate) bits in the range [start, end) in place.
    pub fn flipInplace(self: *Roaring32Bitmap, start: u64, end: u64) void {
        roaring_bitmap_flip_inplace(self.ptr, start, end);
    }

    /// rank gets the rank (number of elements <= value) of a value
    pub fn rank(self: *const Roaring32Bitmap, value: u32) u64 {
        return roaring_bitmap_rank(self.ptr, value);
    }

    /// select gets the element at the given rank (0-indexed) and returns null if rank is out of bounds.
    pub fn select(self: *const Roaring32Bitmap, rank_index: u32) ?u32 {
        var element: u32 = undefined;
        if (roaring_bitmap_select(self.ptr, rank_index, &element)) {
            return element;
        }
        return null;
    }

    /// getIndex gets the index of a value in the bitmap and returns -1 if the value is not present
    pub fn getIndex(self: *const Roaring32Bitmap, value: u32) i64 {
        return roaring_bitmap_get_index(self.ptr, value);
    }

    /// toArray converts the bitmap to a sorted array; the caller owns the returned memory and must free it
    pub fn toArray(self: *const Roaring32Bitmap, allocator: std.mem.Allocator) ![]u32 {
        const card = self.getCardinality();
        const array = try allocator.alloc(u32, card);
        roaring_bitmap_to_uint32_array(self.ptr, array.ptr);
        return array;
    }

    /// runOptimize optimizes the bitmap for better compression and returns true if the result has at least one run container
    pub fn runOptimize(self: *Roaring32Bitmap) bool {
        return roaring_bitmap_run_optimize(self.ptr);
    }

    /// shrinkToFit shrinks memory usage to fit current contents and returns the number of bytes saved
    pub fn shrinkToFit(self: *Roaring32Bitmap) usize {
        return roaring_bitmap_shrink_to_fit(self.ptr);
    }

    /// portableSizeInBytes gets the size in bytes needed to serialize this bitmap (portable format)
    pub fn portableSizeInBytes(self: *const Roaring32Bitmap) usize {
        return roaring_bitmap_portable_size_in_bytes(self.ptr);
    }

    /// portableSerialize serializes the bitmap to a buffer (portable format)
    /// The buffer must be at least portableSizeInBytes() in size and it returns the number of bytes written.
    pub fn portableSerialize(self: *const Roaring32Bitmap, buffer: []u8) usize {
        return roaring_bitmap_portable_serialize(self.ptr, buffer.ptr);
    }

    /// portableDeserialize deserializes a bitmap from a buffer (portable format)
    pub fn portableDeserialize(buffer: []const u8) ?*Roaring32Bitmap {
        const bitmap_ptr = roaring_bitmap_portable_deserialize_safe(buffer.ptr, buffer.len) orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring32Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    /// printf prints the bitmap contents (for debugging)
    pub fn printf(self: *const Roaring32Bitmap) void {
        roaring_bitmap_printf(self.ptr);
    }

    /// Iterator for roaring bitmap values
    pub const Iterator = struct {
        it: roaring_uint32_iterator_t,

        pub fn init(bitmap: *const Roaring32Bitmap) Iterator {
            var it: roaring_uint32_iterator_t = undefined;
            roaring_iterator_init(bitmap.ptr, &it);
            return .{ .it = it };
        }

        /// next gets the next value. Returns null if iteration is complete
        pub fn next(self: *Iterator) ?u32 {
            if (!self.it.has_value) return null;
            const value = self.it.current_value;
            _ = roaring_uint32_iterator_advance(&self.it);
            return value;
        }

        /// hasValue checks if there are more values
        pub fn hasValue(self: *const Iterator) bool {
            return self.it.has_value;
        }

        /// currentValue gets the current value without advancing
        pub fn currentValue(self: *const Iterator) ?u32 {
            if (!self.it.has_value) return null;
            return self.it.current_value;
        }
    };
};

test "basic roaring bitmap operations" {
    const bitmap = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap.deinit();

    try std.testing.expect(bitmap.isEmpty());
    try std.testing.expectEqual(@as(u64, 0), bitmap.getCardinality());

    bitmap.add(1);
    bitmap.add(2);
    bitmap.add(3);

    try std.testing.expect(!bitmap.isEmpty());
    try std.testing.expectEqual(@as(u64, 3), bitmap.getCardinality());
    try std.testing.expect(bitmap.contains(1));
    try std.testing.expect(bitmap.contains(2));
    try std.testing.expect(bitmap.contains(3));
    try std.testing.expect(!bitmap.contains(4));
}

test "roaring bitmap from slice" {
    const values = [_]u32{ 1, 5, 10, 100, 1000 };
    const bitmap = Roaring32Bitmap.initFromSlice(&values) orelse return error.AllocationFailed;
    defer bitmap.deinit();

    try std.testing.expectEqual(@as(u64, 5), bitmap.getCardinality());
    try std.testing.expect(bitmap.contains(1));
    try std.testing.expect(bitmap.contains(5));
    try std.testing.expect(bitmap.contains(1000));
}

test "roaring bitmap union" {
    const bitmap1 = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap1.deinit();

    const bitmap2 = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap2.deinit();

    bitmap1.add(1);
    bitmap1.add(2);
    bitmap1.add(3);

    bitmap2.add(3);
    bitmap2.add(4);
    bitmap2.add(5);

    const result = bitmap1.unionWith(bitmap2) orelse return error.AllocationFailed;
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 5), result.getCardinality());
    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(result.contains(5));
}

test "roaring bitmap intersection" {
    const bitmap1 = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap1.deinit();

    const bitmap2 = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap2.deinit();

    bitmap1.add(1);
    bitmap1.add(2);
    bitmap1.add(3);

    bitmap2.add(2);
    bitmap2.add(3);
    bitmap2.add(4);

    const result = bitmap1.intersectionWith(bitmap2) orelse return error.AllocationFailed;
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 2), result.getCardinality());
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(4));
}

test "roaring bitmap iterator" {
    const bitmap = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap.deinit();

    bitmap.add(1);
    bitmap.add(5);
    bitmap.add(10);

    var it = Roaring32Bitmap.Iterator.init(bitmap);

    try std.testing.expectEqual(@as(?u32, 1), it.next());
    try std.testing.expectEqual(@as(?u32, 5), it.next());
    try std.testing.expectEqual(@as(?u32, 10), it.next());
    try std.testing.expectEqual(@as(?u32, null), it.next());
}

test "roaring bitmap range operations" {
    const bitmap = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap.deinit();

    bitmap.addRange(10, 20);

    try std.testing.expectEqual(@as(u64, 10), bitmap.getCardinality());
    try std.testing.expect(bitmap.contains(10));
    try std.testing.expect(bitmap.contains(19));
    try std.testing.expect(!bitmap.contains(20));
    try std.testing.expect(!bitmap.contains(9));
}

test "roaring bitmap to array" {
    const allocator = std.testing.allocator;

    const bitmap = Roaring32Bitmap.init() orelse return error.AllocationFailed;
    defer bitmap.deinit();

    bitmap.add(1);
    bitmap.add(5);
    bitmap.add(10);

    const array = try bitmap.toArray(allocator);
    defer allocator.free(array);

    try std.testing.expectEqual(@as(usize, 3), array.len);
    try std.testing.expectEqual(@as(u32, 1), array[0]);
    try std.testing.expectEqual(@as(u32, 5), array[1]);
    try std.testing.expectEqual(@as(u32, 10), array[2]);
}

// ============================================================================
// 64-bit Roaring Bitmap Support
// ============================================================================

/// Opaque type for roaring64_bitmap_t
pub const roaring64_bitmap_t = opaque {};

// External C functions from CRoaring for 64-bit bitmaps
extern "c" fn roaring64_bitmap_create() ?*roaring64_bitmap_t;
extern "c" fn roaring64_bitmap_free(r: *const roaring64_bitmap_t) void;
extern "c" fn roaring64_bitmap_add(r: *roaring64_bitmap_t, val: u64) void;
extern "c" fn roaring64_bitmap_contains(r: *const roaring64_bitmap_t, val: u64) bool;
extern "c" fn roaring64_bitmap_get_cardinality(r: *const roaring64_bitmap_t) u64;
extern "c" fn roaring64_bitmap_is_empty(r: *const roaring64_bitmap_t) bool;
extern "c" fn roaring64_bitmap_portable_size_in_bytes(r: *const roaring64_bitmap_t) usize;
extern "c" fn roaring64_bitmap_portable_serialize(r: *const roaring64_bitmap_t, buf: [*]u8) usize;
extern "c" fn roaring64_bitmap_portable_deserialize_safe(buf: [*]const u8, maxbytes: usize) ?*roaring64_bitmap_t;

/// Roaring64Bitmap is a is a compressed bitmap data structure for 64-bit integers.
/// https://github.com/RoaringBitmap/RoaringFormatSpec#extension-for-64-bit-implementations
pub const Roaring64Bitmap = struct {
    ptr: *roaring64_bitmap_t,

    pub fn init() ?*Roaring64Bitmap {
        const bitmap_ptr = roaring64_bitmap_create() orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring64Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }

    pub fn deinit(self: *Roaring64Bitmap) void {
        roaring64_bitmap_free(self.ptr);
        const allocator = std.heap.c_allocator;
        allocator.destroy(self);
    }

    /// add a value to the bitmap
    pub fn add(self: *Roaring64Bitmap, value: u64) void {
        roaring64_bitmap_add(self.ptr, value);
    }

    /// contains checks if a value is present in the bitmap
    pub fn contains(self: *const Roaring64Bitmap, value: u64) bool {
        return roaring64_bitmap_contains(self.ptr, value);
    }

    /// getCardinality gets the number of elements in the bitmap (cardinality)
    pub fn getCardinality(self: *const Roaring64Bitmap) u64 {
        return roaring64_bitmap_get_cardinality(self.ptr);
    }

    /// isEmpty returns true if the bitmap is empty
    pub fn isEmpty(self: *const Roaring64Bitmap) bool {
        return roaring64_bitmap_is_empty(self.ptr);
    }

    /// portableSizeInBytes gets the size in bytes needed to serialize this bitmap (portable format)
    pub fn portableSizeInBytes(self: *const Roaring64Bitmap) usize {
        return roaring64_bitmap_portable_size_in_bytes(self.ptr);
    }

    /// portableSerialize serializes the bitmap to a buffer in a portable format
    /// Buffer must be at least portableSizeInBytes() in size and it returns the number of bytes written
    pub fn portableSerialize(self: *const Roaring64Bitmap, buffer: []u8) usize {
        return roaring64_bitmap_portable_serialize(self.ptr, buffer.ptr);
    }

    /// portableDeserialize deserializes a bitmap from a buffer in the portable format
    pub fn portableDeserialize(buffer: []const u8) ?*Roaring64Bitmap {
        const bitmap_ptr = roaring64_bitmap_portable_deserialize_safe(buffer.ptr, buffer.len) orelse return null;
        const allocator = std.heap.c_allocator;
        const self = allocator.create(Roaring64Bitmap) catch return null;
        self.* = .{ .ptr = bitmap_ptr };
        return self;
    }
};
