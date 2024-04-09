#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::ops::*;
use std::slice;

use std::arch::aarch64::*;

use super::{Arch, Float, Int, Simd};

pub struct Neon;

impl Arch for Neon {
    type f32 = f32x4;
    type u32 = u32x4;
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct f32x4(float32x4_t);

impl Default for f32x4 {
    #[inline(always)]
    fn default() -> f32x4 {
        unsafe { f32x4(vdupq_n_f32(0.0)) }
    }
}

impl From<f32> for f32x4 {
    #[inline(always)]
    fn from(value: f32) -> f32x4 {
        unsafe { f32x4(vdupq_n_f32(value)) }
    }
}

impl Simd for f32x4 {
    type Elem = f32;

    const LANES: usize = 4;

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe { vgetq_lane_f32(self.0, 3) }
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

        unsafe { f32x4(vld1q_f32(slice.as_ptr())) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            vst1q_f32(slice.as_mut_ptr(), self.0);
        }
    }

    #[inline(always)]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn inner(slice: &[f32]) -> f32x4 {
            match slice.len() {
                0 => f32x4(vdupq_n_f32(0.0)),
                1 => f32x4(vsetq_lane_f32(slice[0], vdupq_n_f32(0.0), 0)),
                2 => f32x4(vcombine_f32(vld1_f32(slice.as_ptr()), vdup_n_f32(0.0))),
                3 => f32x4(vcombine_f32(
                    vld1_f32(slice.as_ptr()),
                    vset_lane_f32(slice[2], vdup_n_f32(0.0), 0),
                )),
                _ => f32x4(vld1q_f32(slice.as_ptr())),
            }
        }

        unsafe { inner(slice) }
    }

    #[inline(always)]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn inner(vector: f32x4, slice: &mut [f32]) {
            match slice.len() {
                0 => {}
                1 => {
                    slice[0] = vgetq_lane_f32(vector.0, 0);
                }
                2 => {
                    vst1_f32(slice.as_mut_ptr(), vget_low_f32(vector.0));
                }
                3 => {
                    vst1_f32(slice.as_mut_ptr(), vget_low_f32(vector.0));
                    slice[2] = vgetq_lane_f32(vector.0, 2)
                }
                _ => vst1q_f32(slice.as_mut_ptr(), vector.0),
            }
        }

        unsafe { inner(*self, slice) }
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
        unsafe { f32x4(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { f32x4(vbslq_f32(vcltq_f32(self.0, rhs.0), self.0, rhs.0)) }
    }

    #[inline(always)]
    fn prefix_sum(self) -> Self {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn inner(value: f32x4) -> f32x4 {
            let shifted = vextq_f32(vdupq_n_f32(0.0), value.0, 3);
            let sum1 = vaddq_f32(value.0, shifted);
            let shifted = vextq_f32(vdupq_n_f32(0.0), sum1, 2);
            f32x4(vaddq_f32(sum1, shifted))
        }

        unsafe { inner(self) }
    }
}

impl Add for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { f32x4(vaddq_f32(self.0, rhs.0)) }
    }
}

impl Sub for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe { f32x4(vsubq_f32(self.0, rhs.0)) }
    }
}

impl Mul for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe { f32x4(vmulq_f32(self.0, rhs.0)) }
    }
}

impl Div for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe { f32x4(vdivq_f32(self.0, rhs.0)) }
    }
}

impl Neg for f32x4 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { f32x4(vnegq_f32(self.0)) }
    }
}

impl From<u32x4> for f32x4 {
    #[inline(always)]
    fn from(value: u32x4) -> f32x4 {
        unsafe { f32x4(vcvtq_f32_u32(value.0)) }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct u32x4(uint32x4_t);

impl Default for u32x4 {
    #[inline(always)]
    fn default() -> u32x4 {
        unsafe { u32x4(vdupq_n_u32(0)) }
    }
}

impl From<u32> for u32x4 {
    #[inline(always)]
    fn from(value: u32) -> u32x4 {
        unsafe { u32x4(vdupq_n_u32(value)) }
    }
}

impl Simd for u32x4 {
    type Elem = u32;

    const LANES: usize = 4;

    #[inline(always)]
    fn last(&self) -> Self::Elem {
        unsafe { vgetq_lane_u32(self.0, 3) }
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

        unsafe { u32x4(vld1q_u32(slice.as_ptr())) }
    }

    #[inline(always)]
    fn store(&self, slice: &mut [Self::Elem]) {
        assert!(slice.len() >= Self::LANES);

        unsafe {
            vst1q_u32(slice.as_mut_ptr(), self.0);
        }
    }

    #[inline(always)]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn inner(slice: &[u32]) -> u32x4 {
            match slice.len() {
                0 => u32x4(vdupq_n_u32(0)),
                1 => u32x4(vsetq_lane_u32(slice[0], vdupq_n_u32(0), 0)),
                2 => u32x4(vcombine_u32(vld1_u32(slice.as_ptr()), vdup_n_u32(0))),
                3 => u32x4(vcombine_u32(
                    vld1_u32(slice.as_ptr()),
                    vset_lane_u32(slice[2], vdup_n_u32(0), 0),
                )),
                _ => u32x4(vld1q_u32(slice.as_ptr())),
            }
        }

        unsafe { inner(slice) }
    }

    #[inline(always)]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn inner(vector: u32x4, slice: &mut [u32]) {
            match slice.len() {
                0 => {}
                1 => {
                    slice[0] = vgetq_lane_u32(vector.0, 0);
                }
                2 => {
                    vst1_u32(slice.as_mut_ptr(), vget_low_u32(vector.0));
                }
                3 => {
                    vst1_u32(slice.as_mut_ptr(), vget_low_u32(vector.0));
                    slice[2] = vgetq_lane_u32(vector.0, 2)
                }
                _ => vst1q_u32(slice.as_mut_ptr(), vector.0),
            }
        }

        unsafe { inner(*self, slice) }
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
            u32x4(vshlq_u32(self.0, vdupq_n_s32(shift as i32)))
        }
    }
}

impl Shr<usize> for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        unsafe {
            let shift = rhs & (u32::BITS as usize - 1);
            u32x4(vshlq_u32(self.0, vdupq_n_s32(-(shift as i8) as i32)))
        }
    }
}

impl BitAnd for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vandq_u32(self.0, rhs.0)) }
    }
}

impl BitOr for u32x4 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { u32x4(vorrq_u32(self.0, rhs.0)) }
    }
}

impl From<f32x4> for u32x4 {
    #[inline(always)]
    fn from(value: f32x4) -> u32x4 {
        unsafe { u32x4(vcvtq_u32_f32(value.0)) }
    }
}
