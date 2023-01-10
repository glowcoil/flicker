#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::ops::*;
use std::slice;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Arch, Float, Int, PossibleArch, Simd, Task};

pub struct Avx2;

impl PossibleArch for Avx2 {
    #[inline]
    fn try_specialize<T: Task>() -> Option<fn(T) -> T::Result> {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner<T: Task>(task: T) -> T::Result {
            task.run::<Avx2Impl>()
        }

        #[inline]
        fn run<T: Task>(task: T) -> T::Result {
            unsafe { inner::<T>(task) }
        }

        if is_x86_feature_detected!("avx2") {
            Some(run::<T>)
        } else {
            None
        }
    }
}

#[cfg(target_feature = "avx2")]
use super::SupportedArch;

#[cfg(target_feature = "avx2")]
impl SupportedArch for Avx2 {
    #[inline]
    fn specialize<T: Task>() -> fn(T) -> T::Result {
        T::run::<Avx2Impl>
    }
}

struct Avx2Impl;

impl Arch for Avx2Impl {
    type f32 = f32x8;
    type u32 = u32x8;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct f32x8(__m256);

impl Default for f32x8 {
    #[inline(always)]
    fn default() -> f32x8 {
        unsafe { f32x8(_mm256_setzero_ps()) }
    }
}

impl From<f32> for f32x8 {
    #[inline(always)]
    fn from(value: f32) -> f32x8 {
        unsafe { f32x8(_mm256_set1_ps(value)) }
    }
}

impl Simd for f32x8 {
    type Elem = f32;

    const LANES: usize = 8;

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe {
            let upper = _mm256_extractf128_ps(self.0, 1);
            _mm_cvtss_f32(_mm_shuffle_ps(upper, upper, 0x03))
        }
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

        unsafe { f32x8(_mm256_loadu_ps(slice.as_ptr())) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            _mm256_storeu_ps(slice.as_mut_ptr(), self.0);
        }
    }
}

impl Debug for f32x8 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Float for f32x8 {
    #[inline(always)]
    fn abs(self) -> Self {
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(!(1 << 31)));
            f32x8(_mm256_and_ps(mask, self.0))
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { f32x8(_mm256_min_ps(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn scan_sum(self) -> Self {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn inner(value: f32x8) -> f32x8 {
            // First, perform two separate prefix sums in the upper and lower 4 lanes:
            let shifted = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(value.0), 4));
            let sum1 = _mm256_add_ps(value.0, shifted);
            let shifted = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(sum1), 8));
            let sum2 = _mm256_add_ps(sum1, shifted);

            // Then, carry the total from the lower 4 lanes to the upper 4 lanes:
            let lower = _mm256_castps256_ps128(sum2);
            let total = _mm_shuffle_ps(lower, lower, 0xFF);
            let carry = _mm256_insertf128_ps(_mm256_setzero_ps(), total, 1);
            f32x8(_mm256_add_ps(sum2, carry))
        }

        unsafe { inner(self) }
    }
}

impl Add for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f32x8(_mm256_add_ps(self.0, rhs.0)) }
    }
}

impl Sub for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f32x8(_mm256_sub_ps(self.0, rhs.0)) }
    }
}

impl Mul for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f32x8(_mm256_mul_ps(self.0, rhs.0)) }
    }
}

impl Div for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f32x8(_mm256_div_ps(self.0, rhs.0)) }
    }
}

impl Neg for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f32x8(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
    }
}

impl From<u32x8> for f32x8 {
    #[inline(always)]
    fn from(value: u32x8) -> f32x8 {
        unsafe { f32x8(_mm256_cvtepi32_ps(value.0)) }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
struct u32x8(__m256i);

impl Default for u32x8 {
    #[inline(always)]
    fn default() -> u32x8 {
        unsafe { u32x8(_mm256_setzero_si256()) }
    }
}

impl From<u32> for u32x8 {
    #[inline(always)]
    fn from(value: u32) -> u32x8 {
        unsafe { u32x8(_mm256_set1_epi32(value as i32)) }
    }
}

impl Simd for u32x8 {
    type Elem = u32;

    const LANES: usize = 8;

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe {
            let upper = _mm256_extracti128_si256(self.0, 1);
            _mm_cvtsi128_si32(_mm_shuffle_epi32(upper, 0x03)) as u32
        }
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

        unsafe { u32x8(_mm256_loadu_si256(slice.as_ptr() as *const __m256i)) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            _mm256_storeu_si256(slice.as_mut_ptr() as *mut __m256i, self.0);
        }
    }
}

impl Debug for u32x8 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Int for u32x8 {}

impl Shl<usize> for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        unsafe {
            let shift = rhs & (u32::BITS as usize - 1);
            u32x8(_mm256_sll_epi32(self.0, _mm_cvtsi64_si128(shift as i64)))
        }
    }
}

impl Shr<usize> for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        unsafe {
            let shift = rhs & (u32::BITS as usize - 1);
            u32x8(_mm256_srl_epi32(self.0, _mm_cvtsi64_si128(shift as i64)))
        }
    }
}

impl BitAnd for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_and_si256(self.0, rhs.0)) }
    }
}

impl BitOr for u32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x8(_mm256_or_si256(self.0, rhs.0)) }
    }
}

impl From<f32x8> for u32x8 {
    #[inline(always)]
    fn from(value: f32x8) -> u32x8 {
        unsafe { u32x8(_mm256_cvtps_epi32(value.0)) }
    }
}
