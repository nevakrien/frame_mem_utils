
// ───────────── STACK ALLOC (untyped, bytes) ────────────────────────────
use core::ops::Index;
use core::ops::IndexMut;
use core::slice;
use core::fmt;
use core::fmt::Write;
use crate::refs::RefBox;
use crate::stack::StackVec;
use core::mem::MaybeUninit;

pub struct StackAlloc<'a>(StackVec<'a, u8>);

impl<'lex> StackAlloc<'lex> {
    #[inline]
    pub const fn from_slice(raw: &'lex mut [MaybeUninit<u8>]) -> Self {
        Self(StackVec::from_slice(raw))
    }

    #[inline]
    pub fn alloc<T>(&mut self) -> Option<&'lex mut MaybeUninit<T>> {
        let curr_len = self.0.len();
        let curr_ptr = unsafe { self.0.get_base().add(curr_len) };

        let pad = curr_ptr.align_offset(align_of::<T>());

        let total = pad + size_of::<T>();
        unsafe {
            self.0.alloc(total)?;
            let slot = curr_ptr.add(pad) as *mut MaybeUninit<T>;
            Some(&mut *slot)
        }
    }

    #[inline]
    pub fn save<T>(&mut self,t:T)->Option<RefBox<'lex,T>> {
        self.alloc().map(|x| 
            unsafe{
                RefBox::new(x.write(t))
            }
        )
    }

    #[inline]
    pub fn save_clone<T:?Sized+Clone>(&mut self,t:&T) -> Option<RefBox<'lex,T>> {
        let curr_len = self.0.len();
        let curr_ptr = unsafe { self.0.get_base().add(curr_len) };

        let pad = curr_ptr.align_offset(align_of::<T>());

        let total = pad + core::mem::size_of_val(t);
        unsafe {
            self.0.alloc(total)?;
            let slot = curr_ptr.add(pad) as *mut MaybeUninit<T>;
            Some(RefBox::new((&mut *slot).write(t.clone())))
        }
    }

    #[inline]
    pub fn save_slice<T:?Sized+Clone>(&mut self,t:&[T]) -> Option<RefBox<'lex,[T]>> {
        let curr_len = self.0.len();
        let curr_ptr = unsafe { self.0.get_base().add(curr_len) };

        let pad = curr_ptr.align_offset(align_of::<T>());

        let total = pad + core::mem::size_of_val(t);
        unsafe {
            self.0.alloc(total)?;
            let slot = curr_ptr.add(pad) as *mut MaybeUninit<T>;
            let slot: &mut [MaybeUninit<T>] = core::slice::from_raw_parts_mut(slot,t.len());
            for (s,x) in slot.into_iter().zip(t.iter()){
                s.write(x.clone());
            }
            let ans = core::slice::from_raw_parts_mut(slot.as_mut_ptr() as *mut T,t.len());
            Some(RefBox::new(ans))
        }
    }


    #[inline]
    pub fn check_point(&self) -> usize {
        self.0.len()
    }

    /// # Safety
    /// No references into the region above the checkpoint may still be live.
    #[inline]
    pub unsafe fn goto_checkpoint(&mut self, cp: usize) {
        let to_free = self.0.len() - cp;
        // Everything here is plain bytes, so dropping isn’t required.
        self.0.free(to_free).expect("checkpoint math is wrong");
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ───────────── WRITER (printf-style, returns &str) ─────────────────────
#[must_use]
pub struct StackWriter<'me, 'lex> {
    alloc: &'me mut StackAlloc<'lex>,
    start: usize,
}

impl Write for StackWriter<'_, '_> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.alloc.0.push_slice(s.as_bytes()).ok_or(fmt::Error)
    }
}

impl<'me, 'lex> StackWriter<'me, 'lex> {
    #[inline]
    pub fn new(alloc: &'me mut StackAlloc<'lex>) -> Self {
        let start = alloc.0.len();
        Self { alloc, start }
    }

    #[inline]
    pub fn finish(self) -> &'lex mut str {
        unsafe {
            let start = self.alloc.0.get_base().add(self.start);
            let len = self.alloc.0.len() - self.start;
            let body = core::slice::from_raw_parts_mut(start, len);
            core::str::from_utf8_unchecked_mut(body)
        }
    }

    #[inline]
    pub fn discard(self) {
        unsafe { self.alloc.goto_checkpoint(self.start) }
    }
}

// ───────────── STACK ALLOC (typed) ─────────────────────────────────────
pub struct StackAllocator<'a, T>(StackVec<'a, T>);


impl<'a, T> StackAllocator<'a, T> {
    #[inline]
    pub const fn new(buf: &'a mut [MaybeUninit<T>]) -> Self {
        Self(StackVec::from_slice(buf))
    }

    #[inline]
    pub fn save(&mut self, elem: T) -> Result<&'a mut T, T> {
        if size_of::<T>() == 0 {
            return Ok(unsafe { &mut *core::ptr::dangling_mut() });
        }

        unsafe {
            match self.0.alloc(1) {
                None => Err(elem),
                Some(_) => {
                    let slot = self.0.peek_raw().unwrap_unchecked();
                    slot.write(elem);
                    Ok(&mut *slot)
                }
            }
        }
    }

    #[inline]
    pub fn check_point(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn get(&self, cp: usize) -> Option<&'a [T]> {
        let live = self.0.len() - cp;
        let addr = self.0.peek_many(live)?.as_ptr().addr();
        let p = self.0.peek_raw()?.with_addr(addr);
        unsafe { Some(slice::from_raw_parts(p, live)) }
    }

    #[inline]
    pub fn index_checkpoint(&self, cp: usize) -> &'a [T] {
        self.get(cp)
            .expect("checkpoint math is wrong")
    }

    /// # Safety
    /// No live references into the abandoned tail may survive.
    #[inline]
    pub unsafe fn goto_checkpoint(&mut self, cp: usize) {
        let live = self.0.len() - cp;
        self.0.flush(live).expect("checkpoint math is wrong"); // drop each value
    }

    /// # Safety
    /// this internal stack lets you break all of the allocators assumbtions
    /// this function should only be used while viewing the code for the allocator itself
    #[inline(always)]
    pub unsafe fn get_inner(&mut self) -> &mut StackVec<'a, T> {
        &mut self.0
    }

    #[inline(always)]
    pub fn with_addr(&self, addr: usize) -> *mut T {
        self.0.get_base().with_addr(addr)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T> Index<usize> for StackAllocator<'_, T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.0[i]
    }
}
impl<T> IndexMut<usize> for StackAllocator<'_, T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.0[i]
    }
}

#[cfg(test)]
mod tests {
    
use super::*;
    use crate::stack::make_storage;
    use core::mem::{MaybeUninit, align_of};

    /// Helper: turn the reference we get back into an integer address.
    #[inline]
    fn addr_of<T>(slot: &mut MaybeUninit<T>) -> usize {
        slot as *mut _ as usize
    }

    /// A 1-byte payload that *demands* 32-byte alignment.
    /// (Rust will still round `size_of::<OverAligned>()` up to 32,
    /// so the object is smaller *logically* than its alignment requirement.)
    #[repr(align(32))]
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct OverAligned(u8);

    struct Zst;

    #[test]
    fn stack_alloc_aligment() {
        // 1 KiB is plenty for the test; adjust if your arena requires more.
        let mut backing: [_; 1024] = make_storage();
        // Safety: we hand the arena exclusive access to `backing`.
        let mut arena = StackAlloc::from_slice(&mut backing);

        /* ── plain u16 (align = 2) ─────────────────────────────────── */
        let s1 = arena.alloc::<u16>().expect("u16 should fit");
        let a1 = addr_of(s1);
        assert_eq!(a1 % align_of::<u16>(), 0, "u16 not aligned");

        /* ── overalligned tiny struct (align = 32 > payload) ───────── */
        let s2 = arena.alloc::<OverAligned>().expect("OverAligned");
        let a2 = addr_of(s2);
        assert_eq!(a2 % align_of::<OverAligned>(), 0, "OverAligned mis-aligned");

        *s2 = MaybeUninit::new(OverAligned(2));
        unsafe { assert_eq!(s2.assume_init(), OverAligned(2)) }

        /* ── zero-sized type (size = 0, align = 1) ─────────────────── */
        let s3 = arena.alloc::<Zst>().expect("ZST");
        *s3 = MaybeUninit::new(Zst);

        let a3 = addr_of(s3);
        assert_eq!(a3 % align_of::<()>(), 0);

        /* ── an array with odd size/alignment interplay ────────────── */
        let s4 = arena.alloc::<[u64; 3]>().expect("[u64;3]");
        let a4 = addr_of(s4);
        assert_eq!(a4 % align_of::<[u64; 3]>(), 0, "array mis-aligned");

        /* ── near-exhaustion check: fill what’s left in 8-byte chunks ─ */
        while let Some(_) = arena.alloc::<u64>(){};

        assert!(arena.alloc::<u64>().is_none(), "OOM must remain OOM");
    }

    #[test]
    fn stack_writer_write_and_finish() {
        let mut backing: [_; 1024] = make_storage(); // 1 KiB arena
        let mut arena = StackAlloc::from_slice(&mut backing);

        let mut writer = StackWriter::new(&mut arena);
        write!(writer, "hello").unwrap();
        write!(writer, " world {}", 42).unwrap();

        let result = writer.finish();
        assert_eq!(result, "hello world 42");

        //try and discard stuff see if we underflow
        let mut writer = StackWriter::new(&mut arena);
        write!(writer, "junk").unwrap();
        writer.discard();

        let mut writer = StackWriter::new(&mut arena);
        write!(writer, "finish").unwrap();
        let result2 = writer.finish();
        assert_eq!(result2, "finish");

        //is it still correct?

        assert_eq!(result, "hello world 42");

        // Make sure what we wrote is indeed valid and no extra allocations happened
        let remaining_space = arena.len();
        let used_bytes = 1024 - remaining_space;

        assert!(
            used_bytes >= result.len() + result2.len(),
            "allocator should have used at least result length"
        );
    }

    #[test]
    fn test_stack_allocator_basic() {
        use alloc::boxed::Box;

        let mut storage = [const { MaybeUninit::<Box<i32>>::uninit() }; 8];
        let mut alloc = StackAllocator::new(&mut storage);

        let a = alloc.save(Box::new(10)).unwrap();
        let b = alloc.save(Box::new(20)).unwrap();

        assert_eq!(**a, 10);
        assert_eq!(**b, 20);

        let cp = alloc.check_point();
        let c = alloc.save(Box::new(30)).unwrap();
        assert_eq!(*c, Box::new(30));

        unsafe {
            alloc.goto_checkpoint(cp);
        }

        // allocation after rollback should overwrite 30
        let d = alloc.save(Box::new(99)).unwrap();
        assert_eq!(*d, Box::new(99));
    }

    #[test]
    fn test_stack_alloc_boxes() {
        use alloc::boxed::Box;

        let mut storage = [const { MaybeUninit::uninit() }; 1024];
        let mut alloc = StackAlloc::from_slice(&mut storage);

        let a = alloc.save_slice(&[10,2]).unwrap();
        let b = alloc.save_clone(&Box::new(20)).unwrap();

        assert_eq!(&*a, &[10,2]);
        assert_eq!(**b, 20);

        let cp = alloc.check_point();
        let c = alloc.save(Box::new(30)).unwrap();
        assert_eq!(*c, Box::new(30));

        core::mem::drop(c);
        unsafe {
            alloc.goto_checkpoint(cp);
        }

        // allocation after rollback should overwrite 30
        let d = alloc.save(Box::new(99)).unwrap();
        assert_eq!(*d, Box::new(99));

    }
}