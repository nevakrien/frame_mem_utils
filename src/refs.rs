use core::ops::Deref;
use core::ops::DerefMut;
use alloc::boxed::Box;
use core::marker::PhantomData;
use core::ptr::NonNull;

#[derive(Debug,PartialEq,Eq,Hash)]
pub struct RefBox<'a,T:?Sized>{
    ptr:NonNull<T>,
    _ph:PhantomData<&'a mut T>,
}

unsafe impl<'m,T> Send for RefBox<'m, T> where Box<T> : Send{}
unsafe impl<'m, T> Sync for RefBox<'m, T> where Box<T> : Sync{}


impl<'a, T:?Sized> RefBox<'a,T>{
    pub unsafe fn new(t:&'a mut T)->Self{
        Self{
            ptr:t.into(),
            _ph:PhantomData
        }
    }

    pub fn leak(self)->&'a mut T{
        let ans = unsafe{
            &mut* self.ptr.as_ptr()
        };
        core::mem::forget(self);
        ans
    }
}

impl<T:?Sized> Drop for RefBox<'_, T>{

fn drop(&mut self) { unsafe{
    core::ptr::drop_in_place(self.ptr.as_ptr())
} }
}

impl<T:?Sized> Deref for RefBox<'_,T>{
type Target = T;
fn deref(&self) -> &T { unsafe{&*self.ptr.as_ptr()}}

}
impl<T:?Sized> DerefMut for RefBox<'_,T>{
fn deref_mut(&mut self) -> &mut T { unsafe{&mut *self.ptr.as_ptr()} }
}