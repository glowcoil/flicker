#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::ops::*;
use std::slice;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{Arch, Float, Int, Simd};

pub struct Avx2;

impl Arch for Avx2 {
    type f32 = f32x8;
    type u32 = u32x8;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct f32x8(__m256);

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

    #[inline(always)]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(slice: &[f32]) -> f32x8 {
            match slice.len() {
                0 => f32x8(_mm256_setzero_ps()),
                1 => f32x8(_mm256_setr_m128(
                    _mm_setr_ps(slice[0], 0.0, 0.0, 0.0),
                    _mm_setzero_ps(),
                )),
                2 => f32x8(_mm256_setr_m128(
                    _mm_setr_ps(slice[0], slice[1], 0.0, 0.0),
                    _mm_setzero_ps(),
                )),
                3 => f32x8(_mm256_setr_m128(
                    _mm_setr_ps(slice[0], slice[1], slice[2], 0.0),
                    _mm_setzero_ps(),
                )),
                4 => f32x8(_mm256_setr_m128(
                    _mm_loadu_ps(slice.as_ptr()),
                    _mm_setzero_ps(),
                )),
                5 => f32x8(_mm256_setr_m128(
                    _mm_loadu_ps(slice.as_ptr()),
                    _mm_setr_ps(slice[4], 0.0, 0.0, 0.0),
                )),
                6 => f32x8(_mm256_setr_m128(
                    _mm_loadu_ps(slice.as_ptr()),
                    _mm_setr_ps(slice[4], slice[5], 0.0, 0.0),
                )),
                7 => f32x8(_mm256_setr_m128(
                    _mm_loadu_ps(slice.as_ptr()),
                    _mm_setr_ps(slice[4], slice[5], slice[6], 0.0),
                )),
                _ => f32x8(_mm256_loadu_ps(slice.as_ptr())),
            }
        }

        unsafe { inner(slice) }
    }

    #[inline(always)]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(vector: f32x8, slice: &mut [f32]) {
            match slice.len() {
                0 => {}
                1 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    slice[0] = _mm_cvtss_f32(lower);
                }
                2 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    slice[0] = _mm_cvtss_f32(lower);
                    slice[1] = _mm_cvtss_f32(_mm_shuffle_ps(lower, lower, 0x01));
                }
                3 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    slice[0] = _mm_cvtss_f32(lower);
                    slice[1] = _mm_cvtss_f32(_mm_shuffle_ps(lower, lower, 0x01));
                    slice[2] = _mm_cvtss_f32(_mm_shuffle_ps(lower, lower, 0x02));
                }
                4 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    _mm_storeu_ps(slice.as_mut_ptr(), lower);
                }
                5 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    _mm_storeu_ps(slice.as_mut_ptr(), lower);
                    let upper = _mm256_extractf128_ps(vector.0, 1);
                    slice[4] = _mm_cvtss_f32(upper);
                }
                6 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    _mm_storeu_ps(slice.as_mut_ptr(), lower);
                    let upper = _mm256_extractf128_ps(vector.0, 1);
                    slice[4] = _mm_cvtss_f32(upper);
                    slice[5] = _mm_cvtss_f32(_mm_shuffle_ps(upper, upper, 0x01));
                }
                7 => {
                    let lower = _mm256_castps256_ps128(vector.0);
                    _mm_storeu_ps(slice.as_mut_ptr(), lower);
                    let upper = _mm256_extractf128_ps(vector.0, 1);
                    slice[4] = _mm_cvtss_f32(upper);
                    slice[5] = _mm_cvtss_f32(_mm_shuffle_ps(upper, upper, 0x01));
                    slice[6] = _mm_cvtss_f32(_mm_shuffle_ps(upper, upper, 0x02));
                }
                _ => {
                    _mm256_storeu_ps(slice.as_mut_ptr(), vector.0);
                }
            }
        }

        unsafe { inner(*self, slice) }
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
    fn prefix_sum(self) -> Self {
        #[inline]
        #[target_feature(enable = "avx2")]
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
pub struct u32x8(__m256i);

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

    #[inline(always)]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(slice: &[u32]) -> u32x8 {
            match slice.len() {
                0 => u32x8(_mm256_setzero_si256()),
                1 => u32x8(_mm256_setr_m128i(
                    _mm_setr_epi32(slice[0] as i32, 0, 0, 0),
                    _mm_setzero_si128(),
                )),
                2 => u32x8(_mm256_setr_m128i(
                    _mm_setr_epi32(slice[0] as i32, slice[1] as i32, 0, 0),
                    _mm_setzero_si128(),
                )),
                3 => u32x8(_mm256_setr_m128i(
                    _mm_setr_epi32(slice[0] as i32, slice[1] as i32, slice[2] as i32, 0),
                    _mm_setzero_si128(),
                )),
                4 => u32x8(_mm256_setr_m128i(
                    _mm_loadu_si128(slice.as_ptr() as *const __m128i),
                    _mm_setzero_si128(),
                )),
                5 => u32x8(_mm256_setr_m128i(
                    _mm_loadu_si128(slice.as_ptr() as *const __m128i),
                    _mm_setr_epi32(slice[4] as i32, 0, 0, 0),
                )),
                6 => u32x8(_mm256_setr_m128i(
                    _mm_loadu_si128(slice.as_ptr() as *const __m128i),
                    _mm_setr_epi32(slice[4] as i32, slice[5] as i32, 0, 0),
                )),
                7 => u32x8(_mm256_setr_m128i(
                    _mm_loadu_si128(slice.as_ptr() as *const __m128i),
                    _mm_setr_epi32(slice[4] as i32, slice[5] as i32, slice[6] as i32, 0),
                )),
                _ => u32x8(_mm256_loadu_si256(slice.as_ptr() as *const __m256i)),
            }
        }

        unsafe { inner(slice) }
    }

    #[inline(always)]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn inner(vector: u32x8, slice: &mut [u32]) {
            match slice.len() {
                0 => {}
                1 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    slice[0] = _mm_cvtsi128_si32(lower) as u32;
                }
                2 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    slice[0] = _mm_cvtsi128_si32(lower) as u32;
                    slice[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(lower, 0x01)) as u32;
                }
                3 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    slice[0] = _mm_cvtsi128_si32(lower) as u32;
                    slice[1] = _mm_cvtsi128_si32(_mm_shuffle_epi32(lower, 0x01)) as u32;
                    slice[2] = _mm_cvtsi128_si32(_mm_shuffle_epi32(lower, 0x02)) as u32;
                }
                4 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, lower);
                }
                5 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, lower);
                    let upper = _mm256_extracti128_si256(vector.0, 1);
                    slice[4] = _mm_cvtsi128_si32(upper) as u32;
                }
                6 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, lower);
                    let upper = _mm256_extracti128_si256(vector.0, 1);
                    slice[4] = _mm_cvtsi128_si32(upper) as u32;
                    slice[5] = _mm_cvtsi128_si32(_mm_shuffle_epi32(upper, 0x01)) as u32;
                }
                7 => {
                    let lower = _mm256_castsi256_si128(vector.0);
                    _mm_storeu_si128(slice.as_mut_ptr() as *mut __m128i, lower);
                    let upper = _mm256_extracti128_si256(vector.0, 1);
                    slice[4] = _mm_cvtsi128_si32(upper) as u32;
                    slice[5] = _mm_cvtsi128_si32(_mm_shuffle_epi32(upper, 0x01)) as u32;
                    slice[6] = _mm_cvtsi128_si32(_mm_shuffle_epi32(upper, 0x02)) as u32;
                }
                _ => {
                    _mm256_storeu_si256(slice.as_mut_ptr() as *mut __m256i, vector.0);
                }
            }
        }

        unsafe { inner(*self, slice) }
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
