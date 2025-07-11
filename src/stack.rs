//this module has a lot of raw pointers betterbe explicit
#![allow(clippy::needless_lifetimes)]

use crate::refs::RefBox;
use core::fmt::Write;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::Index;
use core::ops::IndexMut;
use core::ptr;
use core::slice;
use core::slice::from_raw_parts_mut;

/// A helper function to create an uninitialized array for backing storage.
///
/// This is a convenience function to avoid unsafe code in user code.
pub fn make_storage<T, const N: usize>() -> [MaybeUninit<T>; N] {
    // SAFETY: MaybeUninit is safe to use uninitialized.
    unsafe { MaybeUninit::uninit().assume_init() }
}

/// A checkpoint for the stack, allowing to rewind to a previous state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackCheckPoint<T>(*mut T);

/*──────────────────── stack type ───────────────────────*/
/// A down-growing, fixed-capacity LIFO stack backed by a slice.
///
/// `DownStack` manages a region of memory as a stack that grows from a high memory
/// address towards a lower one. This is in contrast to typical stack implementations
/// that grow upwards. It's designed for scenarios where you need a stack but want to
/// avoid heap allocations, such as in embedded systems or performance-critical code.
///
/// The internal pointers are not dereferenced during operations, which allows for
/// creating safe aliases to its memory, enabling advanced, unsafe patterns if needed.
///
/// # Layout (high address at the top):
///
/// ```text
///                    above ┐
///                          ▼
///   [ x x x x ...free... ][ ...dead space ...]
///     ▲     ▲
///     │     └─ head
///     └──────── end (low addr)
/// ```
///
/// # Example
///
/// ```rust
/// use frame_mem_utils::stack::{DownStack, make_storage};
/// use core::mem::MaybeUninit;
///
/// let mut storage: [MaybeUninit<i32>; 4] = make_storage();
/// let mut stack = DownStack::from_slice(&mut storage);
///
/// stack.push(10).unwrap();
/// stack.push(20).unwrap();
///
/// assert_eq!(stack.len(), 2);
/// assert_eq!(stack.pop(), Some(20));
/// assert_eq!(stack.pop(), Some(10));
/// assert!(stack.is_empty());
/// ```
pub struct DownStack<'mem, T> {
    //the fact rust is missing const makes this make less effishent code than i would like...
    above: *mut T, // one-past the *highest* live element
    head: *mut T,  // next pop / current top (lowest live element)
    end: *mut T,   // lowest address in the backing buffer
    _ph: PhantomData<&'mem mut [MaybeUninit<T>]>,
}

unsafe impl<'m, T: Send> Send for DownStack<'m, T> {}
unsafe impl<'m, T: Sync> Sync for DownStack<'m, T> {}

impl<T> Drop for DownStack<'_, T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
    }
}

/*──────────────────── constructors ────────────────────*/
impl<'mem, T> DownStack<'mem, T> {
    /// Creates an empty `DownStack` from a given slice of `MaybeUninit<T>`.
    ///
    /// The stack will use this slice as its backing storage.
    #[inline]
    pub const fn from_slice(buf: &'mem mut [MaybeUninit<T>]) -> Self {
        let end = buf.as_mut_ptr() as *mut T; // low addr
        let above = unsafe { end.add(buf.len()) }; // one-past high addr
        Self {
            above,
            head: above,
            end,
            _ph: PhantomData,
        }
    }

    /// Creates an empty `DownStack` from a raw pointer to a slice of `MaybeUninit<T>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is valid for the lifetime `'mem`.
    #[inline]
    pub const unsafe fn from_slice_raw(buf: *mut [MaybeUninit<T>]) -> Self {
        let end = buf as *mut T; // low addr
        let above = unsafe { end.add((*buf).len()) }; // one-past high addr
        Self {
            above,
            head: above,
            end,
            _ph: PhantomData,
        }
    }

    /// Creates a `DownStack` from a slice of already initialized data.
    ///
    /// The stack will be full and all elements will be considered live.
    #[inline]
    pub const fn new_full(buf: &'mem mut [T]) -> Self
    where
        T: Copy,
    {
        let end = buf.as_mut_ptr(); // low addr
        let above = unsafe { end.add(buf.len()) }; // one-past high addr
        Self {
            above,
            head: end,
            end,
            _ph: PhantomData,
        }
    }

    /// Converts the `DownStack` back into the backing slice.
    #[inline]
    pub fn to_slice(self) -> &'mem mut [MaybeUninit<T>] {
        unsafe {
            let len = self.above.offset_from(self.end) as usize;
            let end = self.end as *mut _;
            core::mem::forget(self);
            from_raw_parts_mut(end, len)
        }
    }

    /// Clones the content of this stack to another `DownStack`.
    ///
    /// The other stack will be cleared before the copy.
    pub fn set_other<'b>(&self, other: &mut DownStack<'b, T>) -> Result<(), usize>
    where
        T: Clone,
    {
        other.flush(other.len());
        for (i, x) in self.peek_all().iter().rev().enumerate() {
            other.push(x.clone()).map_err(|_| i)?;
        }
        Ok(())
    }

    /*──────────────────── invariants ───────────────────────*/
    /// Returns the number of live elements in the stack.
    #[inline]
    pub fn write_index(&self) -> usize {
        unsafe { self.above.offset_from(self.head) as usize }
    }

    /// Returns the number of available slots in the stack.
    #[inline]
    pub fn room_left(&self) -> usize {
        unsafe { self.head.offset_from(self.end) as usize }
    }

    /// **Unchecked**: set depth directly (0 ≤ `idx` ≤ capacity).
    /// # Safety
    /// - the index must be in the stacks range
    /// - this now allows bad reads but non would be executed automatically
    #[inline]
    pub unsafe fn set_write_index(&mut self, idx: usize) {
        unsafe {
            self.head = self.above.sub(idx);
        }
    }

    /// Returns the number of elements in the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.write_index()
    }

    /// Returns `true` if the stack contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a raw pointer to the end of the stack's buffer.
    #[inline(always)]
    pub fn get_end(&self) -> *mut T {
        self.end
    }

    ///this is used to manually set the capacity of the stack
    ///# Safety
    /// obviously this is very unsafe and all the underlying memory must be valid
    /// further it needs to be the same allocation
    #[inline(always)]
    pub unsafe fn set_end(&mut self, end: *mut T) {
        self.end = end;
    }

    /// Returns a raw pointer to the head of the stack.
    #[inline(always)]
    pub fn get_head(&self) -> *mut T {
        self.head
    }

    /// # Safety
    /// if head is set outside the allocation (not including 1 above) subsequent ops would be UB
    /// if head is set below its current possition then reads/writes to these elements from things like pop are UB    
    /// if head is set above its current position skiped items wont be dropped
    #[inline(always)]
    pub unsafe fn set_head(&mut self, head: *mut T) {
        self.head = head
    }

    /// Returns a raw pointer to 1 above the stack's buffer.
    #[inline(always)]
    pub fn get_above(&self) -> *mut T {
        self.above
    }

    ///this is used to manually remove/extend the logical bottom of the stack (phisical top)
    ///# Safety
    /// obviously this is very unsafe and all the underlying memory must be valid
    /// further it needs to be the same allocation
    #[inline(always)]
    pub unsafe fn set_above(&mut self, above: *mut T) {
        self.above = above;
    }

    /*──────────────────── push / pop ───────────────────────*/
    /// Pushes a value onto the top of the stack.
    ///
    /// Returns `Err(v)` if the stack is full.
    #[inline]
    pub fn push(&mut self, v: T) -> Result<(), T> {
        if self.room_left() == 0 {
            return Err(v);
        }
        unsafe {
            self.head = self.head.sub(1);
            self.head.write(v);
        }
        Ok(())
    }

    /// Pops a value from the top of the stack.
    ///
    /// Returns `None` if the stack is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.write_index() == 0 {
            return None;
        }
        unsafe {
            let v = self.head.read();
            self.head = self.head.add(1);
            Some(v)
        }
    }

    /// Pushes an array of `N` elements onto the stack.
    ///
    /// Returns `Err(arr)` if there is not enough room.
    #[inline]
    pub fn push_n<const N: usize>(&mut self, arr: [T; N]) -> Result<(), [T; N]> {
        if self.room_left() < N {
            return Err(arr);
        }
        unsafe {
            self.head = self.head.sub(N);
            (self.head as *mut [T; N]).write(arr);
        }
        Ok(())
    }

    /// Pops an array of `N` elements from the stack.
    ///
    /// Returns `None` if there are not enough elements.
    #[inline]
    pub fn pop_n<const N: usize>(&mut self) -> Option<[T; N]> {
        if self.write_index() < N {
            return None;
        }
        unsafe {
            let out = (self.head as *mut [T; N]).read();
            self.head = self.head.add(N);
            Some(out)
        }
    }

    /// Pushes a slice of elements onto the stack.
    ///
    /// Returns `None` if there is not enough room.
    #[inline]
    pub fn push_slice(&mut self, src: &[T]) -> Option<()>
    where
        T: Clone,
    {
        if self.room_left() < src.len() {
            return None;
        }
        unsafe {
            self.head = self.head.sub(src.len());
            let dst = slice::from_raw_parts_mut(self.head, src.len());
            dst.clone_from_slice(src);
        }
        Some(())
    }

    /// Pops `n` elements from the stack and returns them as a `RefBox<[T]>`.
    ///
    /// Returns `None` if there are not enough elements.
    #[inline]
    pub fn pop_many<'b>(&'b mut self, n: usize) -> Option<RefBox<'b, [T]>> {
        if self.write_index() < n {
            return None;
        }
        unsafe {
            let p = self.head;
            self.head = self.head.add(n);
            Some(RefBox::new(slice::from_raw_parts_mut(p, n)))
        }
    }

    /*──────────────────── allocs ───────────────────────*/
    /// Removes `len` elements from the top of the stack, calling their destructors.
    #[inline]
    pub fn flush(&mut self, len: usize) -> Option<()> {
        if self.write_index() < len {
            return None;
        }
        unsafe {
            //force a drop of these elements
            for x in from_raw_parts_mut(self.head, len) {
                ptr::read(x);
            }

            self.head = self.head.add(len);
            Some(())
        }
    }

    /// Removes `len` elements from the top of the stack without calling their destructors.
    #[inline]
    pub fn free(&mut self, len: usize) -> Option<()> {
        if self.write_index() < len {
            return None;
        }
        unsafe {
            self.head = self.head.add(len);
            Some(())
        }
    }

    /// Allocates `len` uninitialized elements on the stack.
    ///
    /// # Safety
    /// Calling this puts invalid memory on the stack.
    /// Using any read operations on it is UB.
    /// This includes flush; however, free is fine.
    #[inline]
    pub unsafe fn alloc(&mut self, len: usize) -> Option<()> {
        if self.room_left() < len {
            return None;
        }
        self.head = unsafe { self.head.sub(len) };
        Some(())
    }
    /*──────────────────── checkpoint ───────────────────────*/

    /// Creates a checkpoint of the current stack state.
    #[inline]
    pub fn check_point(&self) -> StackCheckPoint<T> {
        StackCheckPoint(self.head)
    }

    /// Rewinds the stack to a previously created checkpoint.
    ///
    /// # Safety
    /// The caller must ensure that no pointers to the freed memory exist.
    #[inline]
    pub unsafe fn goto_checkpoint(&mut self, check_point: StackCheckPoint<T>) {
        self.head = check_point.0;
    }
    /*──────────────────── peek helpers ─────────────────────*/
    /// Returns a reference to the top element of the stack.
    #[inline]
    pub fn peek<'b>(&'b self) -> Option<&'b T> {
        self.peek_n::<1>().map(|a| &a[0])
    }

    /// Returns a reference to the top `N` elements of the stack.
    #[inline]
    pub fn peek_n<'b, const N: usize>(&'b self) -> Option<&'b [T; N]> {
        if self.write_index() < N {
            None
        } else {
            unsafe { Some(&*(self.head as *const [T; N])) }
        }
    }

    /// Returns a slice of the top `n` elements of the stack.
    #[inline]
    pub fn peek_many<'b>(&'b self, n: usize) -> Option<&'b [T]> {
        if self.write_index() < n {
            None
        } else {
            unsafe { Some(slice::from_raw_parts(self.head, n)) }
        }
    }

    /// Returns a slice containing all elements in the stack, from top to bottom.
    #[inline]
    pub fn peek_all<'b>(&'b self) -> &'b [T] {
        unsafe { slice::from_raw_parts(self.head, self.write_index()) }
    }

    /// Returns a mutable reference to the top element of the stack.
    #[inline]
    pub fn peek_mut<'b>(&'b mut self) -> Option<&'b mut T> {
        self.peek_n_mut::<1>().map(|a| &mut a[0])
    }

    /// Returns a mutable reference to the top `N` elements of the stack.
    #[inline]
    pub fn peek_n_mut<'b, const N: usize>(&'b mut self) -> Option<&'b mut [T; N]> {
        if self.write_index() < N {
            None
        } else {
            unsafe { Some(&mut *(self.head as *mut [T; N])) }
        }
    }

    /// Returns a mutable slice of the top `n` elements of the stack.
    #[inline]
    pub fn peek_many_mut<'b>(&'b mut self, n: usize) -> Option<&'b mut [T]> {
        if self.write_index() < n {
            None
        } else {
            unsafe { Some(slice::from_raw_parts_mut(self.head, n)) }
        }
    }

    /// Returns a mutable slice of all elements in the stack.
    #[inline]
    pub fn peek_all_mut<'b>(&'b mut self) -> &'b mut [T] {
        unsafe { slice::from_raw_parts_mut(self.head, self.write_index()) }
    }

    /// Returns a mutable reference to the element at `n` from the top of the stack.
    #[inline]
    pub fn spot<'b>(&'b mut self, n: usize) -> Option<&'b mut T> {
        if self.write_index() <= n {
            None
        } else {
            unsafe { Some(&mut *self.head.add(n)) }
        }
    }

    /// Returns a raw pointer to the element at `n` from the top of the stack.
    #[inline]
    pub fn spot_raw(&self, n: usize) -> Option<*mut T> {
        if self.write_index() <= n {
            None
        } else {
            unsafe { Some(self.head.add(n)) }
        }
    }

    /*──────────────────── complex handeling ─────────────────────*/
    /// Drop `count` items located `skip` elements below the top.
    pub fn drop_inside(&mut self, skip: usize, count: usize) -> Option<()> {
        if skip + count > self.write_index() {
            return None;
        }
        unsafe {
            let p = self.head.add(skip);
            // Explicitly drop the region [p, p+count)
            for x in from_raw_parts_mut(p, count) {
                ptr::read(x);
            }
            // Move the upper part down
            ptr::copy(self.head, self.head.add(count), skip);
            self.head = self.head.add(count);
        }
        Some(())
    }

    /// Splits the stack into a slice of its elements and an empty `DownStack`.
    /// (live slice on the left, empty stack on the right)
    #[inline]
    pub fn split<'b>(&'b mut self) -> (&'b mut [T], DownStack<'b, T>) {
        let live = self.write_index();
        let left = unsafe { slice::from_raw_parts_mut(self.head, live) };
        let right = DownStack {
            above: self.head,
            head: self.head,
            end: self.end,
            _ph: PhantomData,
        };
        (left, right)
    }

    /// Consumes the stack and returns a slice of its elements and an empty `DownStack`.
    #[inline]
    pub fn split_consume(self) -> (&'mem mut [T], DownStack<'mem, T>) {
        let live = self.write_index();
        let left = unsafe { slice::from_raw_parts_mut(self.head, live) };
        let right = DownStack {
            above: self.head,
            head: self.head,
            end: self.end,
            _ph: PhantomData,
        };
        (left, right)
    }

    /// Raw pointer to live slice + empty right-hand stack (no borrow).
    #[inline]
    pub fn split_raw(&mut self) -> (*mut T, DownStack<'_, T>) {
        let ptr_live = self.head;
        let right = DownStack {
            above: self.head,
            head: self.head,
            end: self.end,
            _ph: PhantomData,
        };
        (ptr_live, right)
    }
}
/*──────────────────── iterator ─────────────────────────*/
impl<T> Iterator for DownStack<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.pop()
    }
}

impl<T> Index<usize> for DownStack<'_, T> {
    type Output = T;
    fn index<'a>(&'a self, id: usize) -> &'a T {
        if self.len() <= id {
            panic!("out of bound index");
        }
        unsafe { &*self.above.sub(1 + id) }
    }
}

impl<T> IndexMut<usize> for DownStack<'_, T> {
    fn index_mut<'a>(&'a mut self, id: usize) -> &'a mut T {
        if self.len() <= id {
            panic!("out of bound index");
        }
        unsafe { &mut *self.above.sub(1 + id) }
    }
}

#[test]
fn test_index_and_index_mut_basic() {
    let mut storage = make_storage::<u32, 4>();
    let mut stack = DownStack::from_slice(&mut storage);

    for v in 1..=3 {
        stack.push(v).unwrap();
    } // 1 oldest … 3 newest

    // Read through Index
    assert_eq!(stack[0], 1);
    assert_eq!(stack[1], 2);
    assert_eq!(stack[2], 3);

    // Mutate the middle element
    stack[1] = 42;
    // Pop order is still newest-first
    assert_eq!(stack.pop(), Some(3));
    assert_eq!(stack.pop(), Some(42));
    assert_eq!(stack.pop(), Some(1));
    assert!(stack.pop().is_none());
}

#[test]
fn test_lifo_order() {
    let mut storage = make_storage::<&'static str, 3>();
    let mut stack = DownStack::from_slice(&mut storage);

    stack.push("first").unwrap();
    stack.push("second").unwrap();
    stack.push("third").unwrap();

    assert_eq!(stack.pop(), Some("third"));
    assert_eq!(stack.peek(), Some("second").as_ref());
    assert_eq!(stack.pop(), Some("second"));
    assert_eq!(stack.pop(), Some("first"));
    assert_eq!(stack.pop(), None);
}

#[test]
fn test_peek_n() {
    let mut storage = make_storage::<u32, 5>();
    let mut stack = DownStack::from_slice(&mut storage);

    for v in 1..=5 {
        stack.push(v).unwrap();
    }

    // Top-first order: 5 4 3
    assert_eq!(stack.peek_n::<3>().unwrap(), &[5, 4, 3]);

    assert!(stack.peek_n::<6>().is_none());
    assert!(stack.peek_n::<5>().is_some());

    assert_eq!(stack.pop(), Some(5));
    assert!(stack.peek_n::<3>().is_some());

    assert_eq!(stack.pop(), Some(4));
    assert!(stack.peek_n::<4>().is_none());
}

#[test]
fn test_peek_many() {
    let mut storage = make_storage::<u32, 6>();
    let mut stack = DownStack::from_slice(&mut storage);

    for i in 1..=5 {
        stack.push(i).unwrap();
    }

    assert_eq!(stack.peek_many(3).unwrap(), &[5, 4, 3]);
    assert!(stack.peek_many(6).is_none());
    assert_eq!(stack.peek_many(5).unwrap(), &[5, 4, 3, 2, 1]);

    stack.pop();
    stack.pop();
    assert!(stack.peek_many(4).is_none());
    assert_eq!(stack.peek_many(3).unwrap(), &[3, 2, 1]);
}

#[test]
fn test_push_n_and_pop_n_success() {
    let mut storage = make_storage::<u32, 6>();
    let mut stack = DownStack::from_slice(&mut storage);

    let arr1 = [10, 20];
    let arr2 = [30, 40, 50];

    stack.push_n(arr1).unwrap();
    stack.push_n(arr2).unwrap();

    assert_eq!(stack.pop_n::<3>().unwrap(), [30, 40, 50]);
    assert_eq!(stack.pop_n::<2>().unwrap(), [10, 20]);
    assert!(stack.pop_n::<1>().is_none());
}

#[test]
fn test_push_n_overflow() {
    let mut storage = make_storage::<u32, 4>();
    let mut stack = DownStack::from_slice(&mut storage);

    let ok = [1, 2];
    let fail = [3, 4, 5];

    assert!(stack.push_n(ok).is_ok());
    assert_eq!(stack.push_n(fail), Err(fail));
}

#[test]
fn test_pop_n_underflow() {
    let mut storage = make_storage::<u32, 3>();
    let mut stack = DownStack::from_slice(&mut storage);

    stack.push(1).unwrap();
    assert!(stack.pop_n::<2>().is_none());
    assert!(stack.pop_n::<1>().is_some());
    assert!(stack.pop_n::<1>().is_none());
}

#[test]
fn test_mixed_push_pop_n() {
    let mut storage = make_storage::<u32, 6>();
    let mut stack = DownStack::from_slice(&mut storage);

    stack.push_n([1, 2, 3]).unwrap();
    stack.push(4).unwrap();
    stack.push_n([5, 6]).unwrap();

    assert_eq!(stack.pop_n::<2>(), Some([5, 6]));
    assert_eq!(stack.pop(), Some(4));
    assert_eq!(stack.pop_n::<3>(), Some([1, 2, 3]));
    assert!(stack.pop().is_none());
}

#[test]
fn test_slice_conversion_basic() {
    let mut storage = make_storage::<u32, 4>();
    let mut stack = DownStack::from_slice(&mut storage);

    assert_eq!(stack.pop(), None);

    stack.push(10).unwrap();
    stack.push(20).unwrap();

    let idx = stack.write_index();
    assert_eq!(idx, 2);

    let slice = stack.to_slice();
    let mut stack = DownStack::from_slice(slice);
    unsafe { stack.set_write_index(idx) };

    assert_eq!(stack.pop(), Some(20));
    assert_eq!(stack.pop(), Some(10));
    assert_eq!(stack.pop(), None);
}

#[test]
fn test_split_stack() {
    let mut storage = make_storage::<u32, 6>();
    let mut original = DownStack::from_slice(&mut storage);

    original.push(1).unwrap();
    original.push(2).unwrap();
    original.push(3).unwrap();

    let (left, mut right) = original.split();

    assert_eq!(left, [3, 2, 1]); // top-first
    assert_eq!(right.pop(), None);

    right.push(10).unwrap();
    assert_eq!(right.pop(), Some(10));
}

#[test]
fn test_push_slice_success_and_error() {
    let mut storage = make_storage::<u32, 5>();
    let mut stack = DownStack::from_slice(&mut storage);

    let input1 = [1, 2, 3];
    assert_eq!(stack.push_slice(&input1), Some(()));
    assert_eq!(stack.peek_many(3), Some(&[1, 2, 3][..]));

    assert_eq!(stack.push_slice(&[4, 5, 6]), None);
    stack.push_slice(&[4, 5]).unwrap();

    assert_eq!(stack.write_index(), 5);
    assert!(stack.push_slice(&[99]).is_none());

    stack.pop().unwrap();
    assert!(stack.push_slice(&[99, 66]).is_none());

    stack.pop().unwrap();
    assert!(stack.push_slice(&[99, 66, 11]).is_none());
}

#[test]
fn test_weird_write_error() {
    let mut storage = make_storage::<i64, 6>();
    let mut stack = DownStack::from_slice(&mut storage);

    stack.push_slice(&[2]).unwrap();
    stack.push_n([1]).unwrap();

    stack.push_slice(&[2, 3]).unwrap();
    stack.push_n([2]).unwrap();

    assert!(stack.push_slice(&[1, 2, 3]).is_none());
}

#[test]
fn test_full_usage() {
    let mut data = [10, 20, 30, 40, 50, 60];
    let mut stack = DownStack::new_full(&mut data);

    assert_eq!(stack.write_index(), 6);
    assert_eq!(stack.room_left(), 0);

    // Top item (down-growing stack) is 10
    assert_eq!(stack.peek(), Some(&10));

    assert_eq!(stack.spot(2), Some(&mut 30));

    // Drop (40, 50)
    stack.drop_inside(3, 2).unwrap();
    assert_eq!(stack.room_left(), 2);

    // Pop remaining items in LIFO order
    assert_eq!(stack.pop(), Some(10));
    assert_eq!(stack.pop(), Some(20));
    assert_eq!(stack.room_left(), 4);

    assert_eq!(stack.pop(), Some(30));
    assert_eq!(stack.pop(), Some(60));
    assert_eq!(stack.room_left(), 6);

    assert_eq!(stack.pop(), None);

    stack.push(77).unwrap();
    assert_eq!(stack.peek(), Some(&77));
    stack.push(77).unwrap();

    stack.pop_many(4).ok_or(()).unwrap_err();
    stack.pop_many(2).unwrap();
}

#[test]
fn test_drop_in() {
    let mut data = [10, 20, 30, 40, 50, 60];
    let mut stack = DownStack::new_full(&mut data);

    assert_eq!(stack.write_index(), 6);
    assert_eq!(stack.room_left(), 0);

    // Top item (down-growing stack) is 10
    assert_eq!(stack.peek(), Some(&10));

    // Drop (30, 40, 50)
    stack.drop_inside(2, 3).unwrap();
    assert_eq!(stack.room_left(), 3);

    // Pop remaining items in LIFO order
    assert_eq!(stack.pop(), Some(10));
    assert_eq!(stack.pop(), Some(20));
    assert_eq!(stack.room_left(), 5);

    assert_eq!(stack.pop(), Some(60));
    assert_eq!(stack.room_left(), 6);

    assert_eq!(stack.pop(), None);

    stack.push(77).unwrap();
    assert_eq!(stack.peek(), Some(&77));
    stack.push(77).unwrap();

    stack.pop_many(4).ok_or(()).unwrap_err();
    stack.pop_many(2).unwrap();
}

#[test]
fn test_drop_inside_skip_zero_should_remove_top_items() {
    // stack top-to-bottom: 10 20 30 40
    let mut data = [10, 20, 30, 40];
    let mut stack = DownStack::new_full(&mut data);

    // Ask to drop the top two (10,20)
    stack.drop_inside(0, 2).unwrap();

    // ── Expected ─────────────────────────
    // write_index()            == 2
    // subsequent pops: 30 then 40
    // ── Actual with current impl ────────
    // write_index()            is still 4
    // pops start with 10       ← not removed!

    assert_eq!(stack.write_index(), 2, "top items were not removed");
    assert_eq!(stack.pop(), Some(30));
    assert_eq!(stack.pop(), Some(40));
    assert_eq!(stack.pop(), None);
}

#[test]
fn test_zero_capacity_stack() {
    use core::mem::MaybeUninit;

    let mut storage: [MaybeUninit<u32>; 0] = [];
    let mut stack = DownStack::from_slice(&mut storage);

    assert_eq!(stack.write_index(), 0);
    assert_eq!(stack.room_left(), 0);

    assert!(stack.pop().is_none());
    assert_eq!(stack.push(123), Err(123));

    let (slice, mut stack2) = stack.split();

    assert_eq!(slice.len(), 0);
    assert_eq!(stack2.write_index(), 0);
    assert_eq!(stack2.room_left(), 0);

    assert!(stack2.pop().is_none());
    assert_eq!(stack2.push(123), Err(123));
}

#[test]
fn test_stacref_set_other_copies_all() {
    let mut buf1 = make_storage::<i32, 6>();
    let mut buf2 = make_storage::<i32, 4>();

    let mut src = DownStack::from_slice(&mut buf1);
    let mut dst = DownStack::from_slice(&mut buf2);

    for v in 1..=3 {
        src.push(v).unwrap();
    }

    dst.push(23).unwrap(); //make sure we clear dst

    src.set_other(&mut dst).unwrap();

    assert_eq!(dst.pop(), Some(3));
    assert_eq!(dst.pop(), Some(2));
    assert_eq!(dst.pop(), Some(1));
    assert!(dst.is_empty());

    assert_eq!(src.pop(), Some(3));
    assert_eq!(src.pop(), Some(2));
    assert_eq!(src.pop(), Some(1));
    assert!(src.is_empty());
}

/*──────────────────── StackVec ────────────────────────────────*/

/// A `Vec`-like implementation on a fixed-size buffer that grows upwards.
///
/// `StackVec` provides a familiar `Vec`-like interface for managing a contiguous
/// array, but it operates on a pre-allocated slice of memory instead of the heap.
/// It is a LIFO stack that grows from a low memory address towards a higher one.
///
/// This is useful for creating dynamic-length collections in environments where
/// heap allocation is not available or desirable.
///
/// # Layout (high address at the top):
///
/// ```text
///   base (low addr)
///     │
///     ▼
///   [ item1, item2, ... itemN, ... free space ... ]
///                             ▲
///                             └─ len points here (one-past-end)
/// ```
///
/// # Example
///
/// ```rust
/// use frame_mem_utils::stack::{StackVec, make_storage};
/// use core::mem::MaybeUninit;
///
/// let mut buffer: [MaybeUninit<u8>; 16] = make_storage();
/// let mut vec = StackVec::from_slice(&mut buffer);
///
/// vec.push(10u8).unwrap();
/// vec.push_slice(&[20, 30]).unwrap();
///
/// assert_eq!(vec.len(), 3);
/// assert_eq!(vec.peek_all(), &[10, 20, 30]);
/// assert_eq!(vec.pop(), Some(30));
/// ```
pub struct StackVec<'mem, T> {
    base: *mut T,
    len: usize,
    capacity: usize,
    _ph: PhantomData<&'mem mut [MaybeUninit<T>]>,
}

unsafe impl<'m, T: Send> Send for StackVec<'m, T> {}
unsafe impl<'m, T: Sync> Sync for StackVec<'m, T> {}

/*────────── constructors ──────────*/

impl<'mem, T> StackVec<'mem, T> {
    /// Creates an empty `StackVec` from a given slice of `MaybeUninit<T>`.
    ///
    /// The vector will use this slice as its backing storage.
    pub const fn from_slice(buf: &'mem mut [MaybeUninit<T>]) -> Self {
        Self {
            base: buf.as_mut_ptr() as *mut T,
            len: 0,
            capacity: buf.len(),
            _ph: PhantomData,
        }
    }

    /// Creates a `StackVec` from a slice of already initialized data.
    ///
    /// The vector will be full and all elements will be considered live.
    pub const fn new_full(buf: &'mem mut [T]) -> Self
    where
        T: Copy,
    {
        Self {
            base: buf.as_mut_ptr(),
            len: buf.len(),
            capacity: buf.len(),
            _ph: PhantomData,
        }
    }

    /// Converts the `StackVec` back into the backing slice.
    pub fn to_slice(self) -> &'mem mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.base as *mut _, self.capacity) }
    }

    /// Clones the content of this vector to another `StackVec`.
    ///
    /// The other vector will be cleared before the copy.
    pub fn set_other<'b>(&self, other: &mut StackVec<'b, T>) -> Result<(), usize>
    where
        T: Clone,
    {
        other.flush(other.len());
        for (i, x) in self.peek_all().iter().enumerate() {
            other.push(x.clone()).map_err(|_| i)?;
        }
        Ok(())
    }
}

/*────────── invariants & meta ──────────*/

impl<T> StackVec<'_, T> {
    /// Returns the number of elements in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// this is a weird mix of alloc and free it is mainly used for checkpoints
    /// # Safety
    /// 1. the length must be in bounds
    /// 2. if elements are alloced same as alloc
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns a raw pointer to the base of the vector's buffer.
    #[inline(always)]
    pub fn get_base(&self) -> *mut T {
        self.base
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Returns the number of additional elements the vector can hold.
    #[inline]
    pub fn room_left(&self) -> usize {
        self.capacity - self.len
    }
}

/*────────── push / pop ──────────*/

impl<'me, T> StackVec<'me, T> {
    /// Appends an element to the end of the vector.
    ///
    /// Returns `Err(v)` if the vector is full.
    #[inline]
    pub fn push(&mut self, v: T) -> Result<(), T> {
        if self.len == self.capacity {
            return Err(v);
        }
        unsafe {
            self.base.add(self.len).write(v);
        }
        self.len += 1;
        Ok(())
    }

    /// Removes the last element from the vector and returns it.
    ///
    /// Returns `None` if the vector is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        unsafe { Some(self.base.add(self.len).read()) }
    }

    /*──── bulk helpers ────*/
    /// Appends an array of `N` elements to the end of the vector.
    ///
    /// Returns `Err(arr)` if there is not enough room.
    #[inline]
    pub fn push_n<const N: usize>(&mut self, arr: [T; N]) -> Result<(), [T; N]> {
        if self.room_left() < N {
            return Err(arr);
        }
        unsafe {
            (self.base.add(self.len) as *mut [T; N]).write(arr);
        }
        self.len += N;
        Ok(())
    }

    /// Pops an array of `N` elements from the end of the vector.
    ///
    /// Returns `None` if there are not enough elements.
    #[inline]
    pub fn pop_n<const N: usize>(&mut self) -> Option<[T; N]> {
        if self.len < N {
            return None;
        }
        self.len -= N;
        unsafe { Some((self.base.add(self.len) as *mut [T; N]).read()) }
    }

    /// Pops `n` elements from the vector and returns them as a `RefBox<[T]>`.
    ///
    /// Returns `None` if there are not enough elements.
    #[inline]
    pub fn pop_many<'b>(&'b mut self, n: usize) -> Option<RefBox<'b, [T]>> {
        if self.len < n {
            None
        } else {
            unsafe {
                let base = self.base.add(self.len - n);
                self.len -= n;
                Some(RefBox::new(slice::from_raw_parts_mut(base, n)))
            }
        }
    }

    /// Appends a slice of elements to the end of the vector.
    ///
    /// Returns `None` if there is not enough room.
    #[inline]
    pub fn push_slice(&mut self, src: &[T]) -> Option<()>
    where
        T: Clone,
    {
        if self.room_left() < src.len() {
            return None;
        }
        unsafe {
            let dst = slice::from_raw_parts_mut(self.base.add(self.len), src.len());
            dst.clone_from_slice(src);
        }
        self.len += src.len();
        Some(())
    }

    /*────────── mem managment ──────────*/

    /// reserves `len` uninitialised slots (UB to read them).
    /// # Safety
    /// 1. the elements must eventually be inilized before a drop
    /// 2. no pops or peeks of these slots may happen
    #[inline]
    pub unsafe fn alloc(&mut self, len: usize) -> Option<()> {
        if self.room_left() < len {
            return None;
        }
        self.len += len;
        Some(())
    }

    /// Drops `len` values from the top (destructors *run*).
    pub fn flush(&mut self, len: usize) -> Option<()> {
        if self.len < len {
            return None;
        }
        unsafe {
            for i in 0..len {
                ptr::read(self.base.add(self.len - 1 - i));
            }
        }
        self.len -= len;
        Some(())
    }

    /// Discards `len` values from the top (**no** destructors).
    #[inline]
    pub fn free(&mut self, len: usize) -> Option<()> {
        if self.len < len {
            return None;
        }
        self.len -= len;
        Some(())
    }

    /*────────── peeks & spots ──────────*/

    /// Returns a reference to the last element of the vector.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            unsafe { Some(&*self.base.add(self.len - 1)) }
        }
    }

    /// Returns a slice of the last `n` elements of the vector.
    #[inline]
    pub fn peek_many(&self, n: usize) -> Option<&[T]> {
        if self.len < n {
            None
        } else {
            unsafe {
                // contiguous bottom→top order: oldest … newest
                Some(slice::from_raw_parts(self.base.add(self.len - n), n))
            }
        }
    }

    /// Returns a slice containing all elements in the vector.
    #[inline]
    pub fn peek_all<'b>(&'b self) -> &'b [T] {
        unsafe { slice::from_raw_parts(self.base, self.len) }
    }

    /// Returns a mutable reference to the last element of the vector.
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            unsafe { Some(&mut *self.base.add(self.len - 1)) }
        }
    }

    /// Returns a mutable slice of the last `n` elements of the vector.
    #[inline]
    pub fn peek_many_mut(&mut self, n: usize) -> Option<&mut [T]> {
        if self.len < n {
            None
        } else {
            unsafe {
                // contiguous bottom→top order: oldest … newest
                Some(slice::from_raw_parts_mut(self.base.add(self.len - n), n))
            }
        }
    }

    /// Returns a mutable slice of all elements in the vector.
    #[inline]
    pub fn peek_all_mut<'b>(&'b mut self) -> &'b mut [T] {
        unsafe { slice::from_raw_parts_mut(self.base, self.len) }
    }

    /// Raw pointer to the current top (no lifetime).
    #[inline]
    pub fn peek_raw(&self) -> Option<*mut T> {
        if self.len == 0 {
            None
        } else {
            Some(unsafe { self.base.add(self.len - 1) })
        }
    }

    /// Mutable ref to the *n*-th value below the top (0 == top).
    #[inline]
    pub fn spot(&mut self, n: usize) -> Option<&mut T> {
        if n >= self.len {
            return None;
        }
        unsafe { Some(&mut *self.base.add(self.len - 1 - n)) }
    }

    /// Raw pointer variant (no lifetime).
    #[inline]
    pub fn spot_raw(&self, n: usize) -> Option<*mut T> {
        if n >= self.len {
            return None;
        }
        Some(unsafe { self.base.add(self.len - 1 - n) })
    }

    /// Mutable ref to the *n*-th value from the start (start==0).
    #[inline]
    pub fn get(&self, id: usize) -> Option<&T> {
        if self.len <= id {
            return None;
        }
        unsafe { Some(&*self.base.add(id)) }
    }

    /// Mutable ref to the *n*-th value from the start (start==0).
    #[inline]
    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        if self.len <= id {
            return None;
        }
        unsafe { Some(&mut *self.base.add(id)) }
    }

    /// Pointer to the *n*-th value from the start (start==0).
    #[inline]
    pub fn get_raw(&self, id: usize) -> Option<*mut T> {
        if self.len <= id {
            return None;
        }
        unsafe { Some(self.base.add(id)) }
    }

    /*────────── split ──────────*/

    /// Splits the vector into a slice of its elements and an empty `StackVec`.
    /// (live slice on the left, empty StackVec on the right)
    #[inline]
    pub fn split<'b>(&'b mut self) -> (&'b mut [T], StackVec<'b, T>) {
        let live = self.len;
        let left = unsafe { slice::from_raw_parts_mut(self.base, live) };

        let right = StackVec {
            base: unsafe { self.base.add(live) },
            len: 0,
            capacity: self.capacity - live,
            _ph: PhantomData,
        };
        (left, right)
    }

    /// Consumes the vector and returns a slice of its elements and an empty `StackVec`.
    #[inline]
    pub fn split_consume(self) -> (&'me mut [T], StackVec<'me, T>) {
        let live = self.len;
        let left = unsafe { slice::from_raw_parts_mut(self.base, live) };

        let right = StackVec {
            base: unsafe { self.base.add(live) },
            len: 0,
            capacity: self.capacity - live,
            _ph: PhantomData,
        };
        (left, right)
    }

    /// Raw pointer to live slice + empty right-hand `StackVec` (no borrow).
    #[inline]
    pub fn split_raw<'b>(&'b mut self) -> (*mut T, StackVec<'b, T>) {
        let live = self.len;

        let right = StackVec {
            base: unsafe { self.base.add(live) },
            len: 0,
            capacity: self.capacity - live,
            _ph: PhantomData,
        };
        (self.base, right)
    }
}

/*────────── Index / IndexMut ──────────*/

impl<T> Index<usize> for StackVec<'_, T> {
    type Output = T;
    fn index(&self, id: usize) -> &T {
        if id >= self.len {
            panic!("index out of bounds");
        }
        unsafe { &*self.base.add(id) }
    }
}

impl<T> IndexMut<usize> for StackVec<'_, T> {
    fn index_mut(&mut self, id: usize) -> &mut T {
        if id >= self.len {
            panic!("index out of bounds");
        }
        unsafe { &mut *self.base.add(id) }
    }
}

/*────────── Drop ──────────*/

impl<T> Drop for StackVec<'_, T> {
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.len {
                ptr::drop_in_place(self.base.add(i));
            }
        }
    }
}

/*──────────────────── write ───────────────────────────*/

impl Write for StackVec<'_, u8> {
    fn write_str(&mut self, s: &str) -> Result<(), core::fmt::Error> {
        self.push_slice(s.as_bytes()).ok_or(core::fmt::Error)
    }
}

/*──────────────────── tests ───────────────────────────*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_slice_concat_and_peek() {
        let mut buf = make_storage::<u32, 6>();
        let mut st = StackVec::from_slice(&mut buf);

        st.push_slice(&[10, 11]).unwrap(); // bottom
        st.push_slice(&[20, 21]).unwrap(); // now on top

        // Bottom→top order in memory: 10 11 20 21
        assert_eq!(st.peek_many(4).unwrap(), &[10, 11, 20, 21]);

        // Top value via peek / peek_raw
        assert_eq!(st.peek(), Some(&21));
        unsafe {
            assert_eq!(*st.peek_raw().unwrap(), 21);
        }
    }

    #[test]
    fn alloc_flush_free_cycle() {
        let mut buf = make_storage::<u8, 8>();
        let mut st = StackVec::from_slice(&mut buf);

        // Reserve space for 3 bytes (uninitialised)
        unsafe {
            st.alloc(3).unwrap();
        }
        assert_eq!(st.len(), 3);

        // Overwrite the raw slots properly
        for i in 0..3 {
            unsafe { *st.spot_raw(i).unwrap() = (i as u8) + 1 }
        }

        // Push two fully-initialised items
        st.push_slice(&[100, 101]).unwrap();
        assert_eq!(st.len(), 5);

        // Flush (drop) the two live items
        st.flush(2).unwrap();
        assert_eq!(st.len(), 3);

        // Discard the three raw bytes without drop
        st.free(3).unwrap();
        assert!(st.is_empty());
    }

    #[test]
    fn spot_and_mutate() {
        let mut buf = make_storage::<i32, 4>();
        let mut st = StackVec::from_slice(&mut buf);

        st.push_n([1, 2, 3, 4]).unwrap(); // top == 4
        *st.spot(1).unwrap() = 99; // change the 3

        assert_eq!(st.pop(), Some(4));
        assert_eq!(st.pop(), Some(99));
    }

    #[test]
    fn stackvec_set_other_copies_all() {
        let mut buf1 = make_storage::<i32, 4>();
        let mut buf2 = make_storage::<i32, 6>();

        let mut src = StackVec::from_slice(&mut buf1);
        let mut dst = StackVec::from_slice(&mut buf2);

        for v in 1..=3 {
            src.push(v).unwrap();
        }

        dst.push(23).unwrap(); //make sure we clear dst

        src.set_other(&mut dst).unwrap();

        assert_eq!(dst.pop(), Some(3));
        assert_eq!(dst.pop(), Some(2));
        assert_eq!(dst.pop(), Some(1));
        assert!(dst.is_empty());

        assert_eq!(src.pop(), Some(3));
        assert_eq!(src.pop(), Some(2));
        assert_eq!(src.pop(), Some(1));
        assert!(src.is_empty());
    }

    #[test]
    fn split_stackvec() {
        let mut storage = make_storage::<u32, 6>();
        let mut original = StackVec::from_slice(&mut storage);

        original.push(1).unwrap();
        original.push(2).unwrap();
        original.push(3).unwrap();

        let (left, mut right) = original.split();

        assert_eq!(left, [1, 2, 3]); // bottom→top order in memory
        assert_eq!(right.pop(), None);

        right.push(10).unwrap();
        assert_eq!(right.pop(), Some(10));
    }
}
