const std = @import("std");
const rbitz = @import("rbitz");

pub fn main() !void {
    const print = std.debug.print;

    print("=== Roaring Bitmap Example ===\n\n", .{});

    const bitmap1 = rbitz.Roaring64Bitmap.init() orelse {
        print("Failed to allocate bitmap1\n", .{});
        return error.AllocationFailed;
    };
    defer bitmap1.deinit();

    const bitmap2 = rbitz.Roaring64Bitmap.init() orelse {
        print("Failed to allocate bitmap2\n", .{});
        return error.AllocationFailed;
    };
    defer bitmap2.deinit();

    // Add some values to bitmap1
    print("Adding values 1, 2, 3, 100, 1000 to bitmap1...\n", .{});
    bitmap1.add(1);
    bitmap1.add(2);
    bitmap1.add(3);
    bitmap1.add(100);
    bitmap1.add(1000);

    // Add some values to bitmap2
    print("Adding values 2, 3, 4, 50, 1000 to bitmap2...\n", .{});
    bitmap2.add(2);
    bitmap2.add(3);
    bitmap2.add(4);
    bitmap2.add(50);
    bitmap2.add(1000);

    // Show cardinality
    print("\nBitmap1 cardinality: {}\n", .{bitmap1.getCardinality()});
    print("Bitmap2 cardinality: {}\n", .{bitmap2.getCardinality()});

    // Test contains
    print("\nBitmap1 contains 100: {}\n", .{bitmap1.contains(100)});
    print("Bitmap1 contains 50: {}\n", .{bitmap1.contains(50)});
    print("Bitmap2 contains 50: {}\n", .{bitmap2.contains(50)});

    // Test isEmpty
    const empty_bitmap = rbitz.Roaring64Bitmap.init() orelse {
        print("Failed to allocate empty_bitmap\n", .{});
        return error.AllocationFailed;
    };
    defer empty_bitmap.deinit();
    print("\nEmpty bitmap is empty: {}\n", .{empty_bitmap.isEmpty()});
    print("Bitmap1 is empty: {}\n", .{bitmap1.isEmpty()});

    // Test serialization
    const allocator = std.heap.page_allocator;
    const serialized_size = bitmap1.portableSizeInBytes();
    print("\nBitmap1 serialized size: {} bytes\n", .{serialized_size});

    const buffer = try allocator.alloc(u8, serialized_size);
    defer allocator.free(buffer);

    const bytes_written = bitmap1.portableSerialize(buffer);
    print("Bytes written: {}\n", .{bytes_written});

    // Test deserialization
    const deserialized_bitmap = rbitz.Roaring64Bitmap.portableDeserialize(buffer) orelse {
        print("Failed to deserialize bitmap\n", .{});
        return error.DeserializationFailed;
    };
    defer deserialized_bitmap.deinit();

    print("Deserialized bitmap cardinality: {}\n", .{deserialized_bitmap.getCardinality()});
    print("Deserialized bitmap contains 100: {}\n", .{deserialized_bitmap.contains(100)});
    print("Deserialized bitmap contains 999: {}\n", .{deserialized_bitmap.contains(999)});

    print("\n=== Example Complete ===\n", .{});
}
