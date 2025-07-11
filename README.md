# frame_mem_utils

This crate provides methods to work with memory allocation without directly calling heap methods. This allows making arenas entirely on the stack, and it is up to you to decide where the memory would come from.

All types used are actively avoiding dereferencing their internal pointers which allows making references to them elsewhere, this allows for some fairly neat things with unsafe code.

## Usage

Here are a couple of examples demonstrating how to use `frame_mem_utils`.

### Saving Slices

You can use `StackAlloc` to save slices of data of any type that is `Clone`. This is useful for storing variable-sized data in a stack-based arena.

```rust
use frame_mem_utils::allocs::StackAlloc;
use frame_mem_utils::stack::make_storage;
use core::mem::MaybeUninit;

// 1. Create a backing memory buffer on the stack.
// `make_storage` is a convenience function to create an uninitialized array.
let mut buffer: [MaybeUninit<u8>; 1024] = make_storage();

// 2. Create a StackAlloc arena from the buffer.
let mut arena = StackAlloc::from_slice(&mut buffer);

// 3. Save a slice of data into the arena.
let saved_slice = arena.save(vec![1u32, 2, 3, 4]).expect("Failed to save slice");

// The saved vec is a `RefBox`, which acts like a Box<Vec> and would be cleaned up.
assert_eq!(&*saved_slice, &[1, 2, 3, 4]);

// You can also save other data types, like a string slice.
let my_string = "hello world";
let saved_string_slice = arena.save_slice(my_string.as_bytes()).expect("Failed to save string slice");

assert_eq!(&*saved_string_slice, my_string.as_bytes());
```

### Writing Formatted Strings

For building strings, `StackWriter` provides an efficient way to write formatted text directly into the arena, similar to `std::fmt::Write`.

```rust
use frame_mem_utils::allocs::{StackAlloc, StackWriter};
use frame_mem_utils::stack::make_storage;
use core::mem::MaybeUninit;
use core::fmt::Write;

// 1. Create a backing memory buffer.
let mut buffer: [MaybeUninit<u8>; 1024] = make_storage();

// 2. Create a StackAlloc arena.
let mut arena = StackAlloc::from_slice(&mut buffer);

// 3. Create a writer.
let mut writer = StackWriter::new(&mut arena);

// 4. Write formatted strings into the arena.
write!(writer, "Hello, {}! The answer is {}.", "world", 42).unwrap();
let formatted_str = writer.finish();

assert_eq!(formatted_str, "Hello, world! The answer is 42.");

// The arena can still be used for other allocations.
let data = arena.save(12345u64).expect("Failed to save integer");
assert_eq!(*data, 12345);
```

### Using `StackVec`

`StackVec` provides a `Vec`-like API on a fixed-size buffer, growing upwards. It's useful when you need a standard vector-like structure without heap allocation.

```rust
use frame_mem_utils::stack::{StackVec, make_storage};
use core::mem::MaybeUninit;

// 1. Create a backing memory buffer on the stack.
let mut buffer: [MaybeUninit<u8>; 1024] = make_storage();

// 2. Create a StackVec from the buffer.
let mut vec = StackVec::from_slice(&mut buffer);

// 3. Push some values.
vec.push(10u8).unwrap();
vec.push(20).unwrap();
vec.push_slice(&[30, 40, 50]).unwrap();

// 4. Access elements like a slice.
assert_eq!(vec.peek_all(), &[10, 20, 30, 40, 50]);
assert_eq!(vec[1], 20);

// 5. Pop values off the end.
assert_eq!(vec.pop(), Some(50));
assert_eq!(&*vec.pop_many(2).unwrap(), &[30,40]);
assert_eq!(vec.len(), 2);
```