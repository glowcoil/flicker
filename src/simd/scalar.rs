#![allow(non_camel_case_types)]

use std::fmt::{self, Debug};
use std::num::Wrapping;
use std::ops::*;
use std::slice;

use super::{Arch, Float, Int, Simd};

pub struct Scalar;

impl Arch for Scalar {
    type f32 = f32x1;
    type u32 = u32x1;
}

#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct f32x1(f32);

impl From<f32> for f32x1 {
    #[inline]
    fn from(value: f32) -> f32x1 {
        f32x1(value)
    }
}

impl Simd for f32x1 {
    type Elem = f32;

    const LANES: usize = 1;

    #[inline]
    fn last(&self) -> Self::Elem {
        self.0
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        slice::from_ref(&self.0)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        slice::from_mut(&mut self.0)
    }

    #[inline]
    fn load(slice: &[Self::Elem]) -> Self {
        f32x1(slice[0])
    }

    #[inline]
    fn store(&self, slice: &mut [Self::Elem]) {
        slice[0] = self.0;
    }

    #[inline]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        f32x1(slice.first().copied().unwrap_or(0.0))
    }

    #[inline]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        if let Some(first) = slice.first_mut() {
            *first = self.0;
        }
    }
}

impl Debug for f32x1 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Float for f32x1 {
    #[inline]
    fn abs(self) -> Self {
        f32x1(self.0.abs())
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        f32x1(if self.0 < rhs.0 { self.0 } else { rhs.0 })
    }

    #[inline]
    fn prefix_sum(self) -> Self {
        self
    }
}

impl Add for f32x1 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        f32x1(self.0 + rhs.0)
    }
}

impl Sub for f32x1 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        f32x1(self.0 - rhs.0)
    }
}

impl Mul for f32x1 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        f32x1(self.0 * rhs.0)
    }
}

impl Div for f32x1 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        f32x1(self.0 / rhs.0)
    }
}

impl Neg for f32x1 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        f32x1(-self.0)
    }
}

impl From<u32x1> for f32x1 {
    fn from(value: u32x1) -> f32x1 {
        f32x1(value.0 .0 as f32)
    }
}

#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct u32x1(Wrapping<u32>);

impl From<u32> for u32x1 {
    #[inline]
    fn from(value: u32) -> u32x1 {
        u32x1(Wrapping(value))
    }
}

impl Simd for u32x1 {
    type Elem = u32;

    const LANES: usize = 1;

    #[inline]
    fn last(&self) -> Self::Elem {
        self.0 .0
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Elem] {
        slice::from_ref(&self.0 .0)
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        slice::from_mut(&mut self.0 .0)
    }

    #[inline]
    fn load(slice: &[Self::Elem]) -> Self {
        u32x1(Wrapping(slice[0]))
    }

    #[inline]
    fn store(&self, slice: &mut [Self::Elem]) {
        slice[0] = self.0 .0;
    }

    #[inline]
    fn load_partial(slice: &[Self::Elem]) -> Self {
        u32x1(Wrapping(slice.first().copied().unwrap_or(0)))
    }

    #[inline]
    fn store_partial(&self, slice: &mut [Self::Elem]) {
        if let Some(first) = slice.first_mut() {
            *first = self.0 .0;
        }
    }
}

impl Debug for u32x1 {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), fmt)
    }
}

impl Int for u32x1 {}

impl Shl<usize> for u32x1 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self {
        u32x1(self.0 << rhs)
    }
}

impl Shr<usize> for u32x1 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self {
        u32x1(self.0 >> rhs)
    }
}

impl BitAnd for u32x1 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        u32x1(self.0 & rhs.0)
    }
}

impl BitOr for u32x1 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        u32x1(self.0 | rhs.0)
    }
}

impl From<f32x1> for u32x1 {
    fn from(value: f32x1) -> u32x1 {
        u32x1(Wrapping(value.0 as u32))
    }
}
