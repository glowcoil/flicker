#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::ops::*;
use std::slice;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Arch, Float, Int, PossibleArch, Simd, SupportedArch, Task};

pub struct Sse2;

impl PossibleArch for Sse2 {
    #[inline]
    fn try_specialize<T: Task>() -> Option<fn(T) -> T::Result> {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner<T: Task>(task: T) -> T::Result {
            task.run::<Sse2Impl>()
        }

        #[inline]
        fn run<T: Task>(task: T) -> T::Result {
            unsafe { inner::<T>(task) }
        }

        if is_x86_feature_detected!("sse2") {
            Some(run::<T>)
        } else {
            None
        }
    }
}

#[cfg(target_feature = "sse2")]
impl SupportedArch for Sse2 {
    #[inline]
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        T::run::<Sse2Impl>
    }
}

struct Sse2Impl;

impl Arch for Sse2Impl {
    type f32 = f32x4;
    type u32 = u32x4;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct f32x4(__m128);

impl Default for f32x4 {
    #[inline(always)]
    fn default() -> f32x4 {
        unsafe { f32x4(_mm_setzero_ps()) }
    }
}

impl Simd for f32x4 {
    type Elem = f32;

    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: Self::Elem) -> Self {
        unsafe { f32x4(_mm_set1_ps(value)) }
    }

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe { _mm_cvtss_f32(_mm_shuffle_ps(self.0, self.0, 0x03)) }
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        unsafe { slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        unsafe { slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES) }
    }

    #[inline(always)]
    fn load(slice: &[Self::Elem]) -> Self {
        assert!(slice.len() >= Self::LANES);

        unsafe { f32x4(_mm_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            _mm_storeu_ps(slice.as_mut_ptr(), self.0);
        }
    }
}

impl Debug for f32x4 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Float for f32x4 {
    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let mask = _mm_castsi128_ps(_mm_set1_epi32(!(1 << 31)));
            f32x4(_mm_and_ps(mask, self.0))
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { f32x4(_mm_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn scan_sum(self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(value: f32x4) -> f32x4 {
            let shifted = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(value.0), 4));
            let sum1 = _mm_add_ps(value.0, shifted);
            let shifted = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(sum1), 8));
            f32x4(_mm_add_ps(sum1, shifted))
        }

        unsafe { inner(self) }
    }
}

impl Add for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f32x4(_mm_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f32x4(_mm_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f32x4(_mm_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f32x4(_mm_div_ps(self.0, rhs.0)) }
    }
}

impl Neg for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f32x4(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))) }
    }
}

impl From<u32x4> for f32x4 {
    #[inline(always)]
    fn from(value: u32x4) -> f32x4 {
        unsafe { f32x4(_mm_cvtepi32_ps(value.0)) }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct u32x4(__m128i);

impl Default for u32x4 {
    #[inline(always)]
    fn default() -> u32x4 {
        unsafe { u32x4(_mm_setzero_si128()) }
    }
}

impl Simd for u32x4 {
    type Elem = u32;

    const LANES: usize = 4;

    #[inline(always)]
    fn splat(value: Self::Elem) -> Self {
        unsafe { u32x4(_mm_set1_epi32(value as i32)) }
    }

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe { _mm_cvtsi128_si32(_mm_shuffle_epi32(self.0, 0x03)) as u32 }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[Self::Elem] {
        unsafe { slice::from_raw_parts(self as *const Self as *const Self::Elem, Self::LANES) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        unsafe { slice::from_raw_parts_mut(self as *mut Self as *mut Self::Elem, Self::LANES) }
    }

    #[inline(always)]
    fn load(slice: &[Self::Elem]) -> Self {
        assert!(slice.len() >= Self::LANES);

        unsafe { u32x4(_mm_loadu_si128(slice.as_ptr() as *const __m128i)) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, self.0);
        }
    }
}

impl Debug for u32x4 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Int for u32x4 {}

impl Shl<usize> for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        unsafe {
            let shift = rhs & (u32::BITS as usize - 1);
            u32x4(_mm_sll_epi32(self.0, _mm_cvtsi64_si128(shift as i64)))
        }
    }
}

impl Shr<usize> for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        unsafe {
            let shift = rhs & (u32::BITS as usize - 1);
            u32x4(_mm_srl_epi32(self.0, _mm_cvtsi64_si128(shift as i64)))
        }
    }
}

impl BitAnd for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(_mm_and_si128(self.0, rhs.0)) }
    }
}

impl BitOr for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(_mm_or_si128(self.0, rhs.0)) }
    }
}

impl From<f32x4> for u32x4 {
    #[inline(always)]
    fn from(value: f32x4) -> u32x4 {
        unsafe { u32x4(_mm_cvtps_epi32(value.0)) }
    }
}
