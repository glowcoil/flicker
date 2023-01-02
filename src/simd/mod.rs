mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;

use std::fmt::Debug;
use std::ops::*;

pub use scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use avx2::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use sse2::*;

#[allow(non_camel_case_types)]
pub trait Arch {
    type f32: Simd<Elem = f32> + Float + From<Self::u32>;
    type u32: Simd<Elem = u32> + Int + From<Self::f32>;
}

pub trait Simd: Copy + Clone + Debug + Default + Send + Sync + Sized {
    type Elem;

    const LANES: usize;

    fn splat(value: Self::Elem) -> Self;
    fn as_slice(&self) -> &[Self::Elem];
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
    fn from_slice(slice: &[Self::Elem]) -> Self;
    fn write_to_slice(&self, slice: &mut [Self::Elem]);
}

pub trait Float: Sized
where
    Self: Add<Output = Self>,
    Self: Sub<Output = Self>,
    Self: Mul<Output = Self>,
    Self: Div<Output = Self>,
    Self: Neg<Output = Self>,
{
    fn scan_sum(self) -> Self;
}

pub trait Int: Sized
where
    Self: Shl<usize, Output = Self>,
    Self: Shr<usize, Output = Self>,
    Self: BitAnd<Output = Self>,
    Self: BitOr<Output = Self>,
{
}

pub trait PossibleArch {
    fn try_specialize<T: Task>() -> Option<fn(T) -> T::Result>;
}

pub trait SupportedArch {
    fn specialize<T: Task>() -> fn(T) -> T::Result;
}

pub trait Task {
    type Result;

    fn run<A: Arch>(self) -> Self::Result;
}
