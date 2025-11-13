.PHONY: test
test:
	zig build test

.PHONY: clean
clean:
	rm -rf .zig-cache zig-out

.PHONY: build
build:
	zig build

