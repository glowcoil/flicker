mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
mod avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "sse2")]
mod sse2;

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
mod neon;

use std::fmt::Debug;
use std::ops::*;

pub use scalar::Scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx2")]
pub use avx2::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "sse2")]
pub use sse2::*;

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
pub use neon::*;

#[allow(non_camel_case_types)]
pub trait Arch {
    type f32: Simd<Elem = f32> + Float + From<Self::u32>;
    type u32: Simd<Elem = u32> + Int + From<Self::f32>;
}

pub trait Simd: Copy + Clone + Debug + Default + Send + Sync + Sized
where
    Self: From<Self::Elem>,
{
    type Elem;

    const LANES: usize;

    fn last(&self) -> Self::Elem;
    fn as_slice(&self) -> &[Self::Elem];
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
    fn load(slice: &[Self::Elem]) -> Self;
    fn store(&self, slice: &mut [Self::Elem]);
    fn load_partial(slice: &[Self::Elem]) -> Self;
    fn store_partial(&self, slice: &mut [Self::Elem]);
}

pub trait Float: Sized
where
    Self: Add<Output = Self>,
    Self: Sub<Output = Self>,
    Self: Mul<Output = Self>,
    Self: Div<Output = Self>,
    Self: Neg<Output = Self>,
{
    fn abs(self) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn prefix_sum(self) -> Self;
}

pub trait Int: Sized
where
    Self: Shl<usize, Output = Self>,
    Self: Shr<usize, Output = Self>,
    Self: BitAnd<Output = Self>,
    Self: BitOr<Output = Self>,
{
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use super::*;

    fn test_f32<A: Arch>() {
        let values = [
            -3.0,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            -f32::EPSILON,
            f32::EPSILON,
            f32::MIN,
            f32::MAX,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ]
        .into_iter()
        .cycle()
        .take(64)
        .collect::<Vec<f32>>();

        for chunk_a in values.chunks(A::f32::LANES) {
            for chunk_b in values.chunks(A::f32::LANES) {
                let result = A::f32::load(chunk_a) + A::f32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a + b;
                    assert!(
                        correct.to_bits() == c.to_bits(),
                        "expected {a} + {b} == {correct}, got {c}"
                    );
                }

                let result = A::f32::load(chunk_a) - A::f32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a - b;
                    assert!(
                        correct.to_bits() == c.to_bits(),
                        "expected {a} - {b} == {correct}, got {c}"
                    );
                }

                let result = A::f32::load(chunk_a) * A::f32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a * b;
                    assert!(
                        correct.to_bits() == c.to_bits(),
                        "expected {a} * {b} == {correct}, got {c}"
                    );
                }

                let result = A::f32::load(chunk_a) / A::f32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a / b;
                    assert!(
                        correct.to_bits() == c.to_bits(),
                        "expected {a} / {b} == {correct}, got {c}"
                    );
                }

                let result = A::f32::load(chunk_a).min(A::f32::load(chunk_b));
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = if a < b { a } else { b };
                    assert!(
                        correct.to_bits() == c.to_bits(),
                        "expected {a}.min({b}) == {correct}, got {c}"
                    );
                }
            }
        }

        for chunk in values.chunks(A::f32::LANES) {
            let result = A::f32::load(chunk).last();
            let correct = *chunk.last().unwrap();
            assert!(
                correct.to_bits() == result.to_bits(),
                "expected {chunk:?}.last() == {correct}, got {result}"
            );

            let result = -A::f32::load(chunk);
            for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                let correct = -a;
                assert!(
                    correct.to_bits() == b.to_bits(),
                    "expected -{a} == {correct}, got {b}"
                );
            }

            let result = A::f32::load(chunk).abs();
            for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                let correct = a.abs();
                assert!(
                    correct.to_bits() == b.to_bits(),
                    "expected {a}.abs() == {correct}, got {b}"
                );
            }

            let result = A::u32::from(A::f32::load(chunk));
            for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                let correct = a as u32;
                assert!(correct == b, "expected {a} as u32 == {correct}, got {b}");
            }
        }

        // Test for prefix_sum. Since different backends use different reduction trees, we can't
        // expect bit equality in the face of rounding or edge cases like NaN or infinity, so
        // just test on exactly representable integers for now.

        let values = (0..64).map(|x| x as f32).collect::<Vec<f32>>();

        for chunk in values.chunks(A::f32::LANES) {
            let result = A::f32::load(chunk).prefix_sum();
            let mut correct = A::f32::load(chunk);
            let mut accum = 0.0;
            for x in correct.as_mut_slice() {
                accum += *x;
                *x = accum;
            }
            let equal =
                result.as_slice().iter().zip(correct.as_slice().iter()).all(|(a, b)| a == b);
            assert!(equal, "prefix_sum() failed\n   input: {chunk:?}\nexpected: {correct:?}\n     got: {result:?}");
        }

        for lanes in 0..A::f32::LANES {
            let mut arr = vec![1.0; A::f32::LANES];
            A::f32::load_partial(&arr[..lanes]).store(&mut arr);
            assert_eq!(arr.iter().sum::<f32>() as usize, lanes);
        }

        for lanes in 0..A::f32::LANES {
            let mut arr = vec![0.0; A::f32::LANES];
            A::f32::from(1.0).store_partial(&mut arr[..lanes]);
            assert_eq!(arr.iter().sum::<f32>() as usize, lanes);
        }
    }

    #[test]
    fn scalar_f32() {
        test_f32::<Scalar>();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(target_feature = "sse2")]
    #[test]
    fn sse2_f32() {
        test_f32::<Sse2>();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(target_feature = "avx2")]
    #[test]
    fn avx2_f32() {
        test_f32::<Avx2>();
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    #[test]
    fn neon_f32() {
        test_f32::<Neon>();
    }

    fn test_u32<A: Arch>() {
        let values = (0..64).collect::<Vec<u32>>();

        for chunk in values.chunks(A::u32::LANES) {
            for shift in 0..u32::BITS as usize * 2 {
                let result = A::u32::load(chunk) << shift;
                for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                    let correct = (Wrapping(a) << shift).0;
                    assert!(
                        correct == b,
                        "expected {a} << {shift} == {correct}, got {b}"
                    );
                }

                let result = A::u32::load(chunk) >> shift;
                for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                    let correct = (Wrapping(a) >> shift).0;
                    assert!(
                        correct == b,
                        "expected {a} >> {shift} == {correct}, got {b}"
                    );
                }
            }
        }

        for chunk_a in values.chunks(A::u32::LANES) {
            for chunk_b in values.chunks(A::u32::LANES) {
                let result = A::u32::load(chunk_a) & A::u32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a & b;
                    assert!(correct == c, "expected {a} & {b} == {correct}, got {c}");
                }

                let result = A::u32::load(chunk_a) | A::u32::load(chunk_b);
                for ((&a, &b), &c) in
                    chunk_a.iter().zip(chunk_b.iter()).zip(result.as_slice().iter())
                {
                    let correct = a | b;
                    assert!(correct == c, "expected {a} | {b} == {correct}, got {c}");
                }
            }
        }

        for chunk in values.chunks(A::u32::LANES) {
            let result = A::u32::load(chunk).last();
            let correct = *chunk.last().unwrap();
            assert!(
                correct == result,
                "expected {chunk:?}.last() == {correct}, got {result}"
            );

            let result = A::f32::from(A::u32::load(chunk));
            for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                let correct = a as f32;
                assert!(correct == b, "expected {a} as f32 == {correct}, got {b}");
            }
        }

        for lanes in 0..A::u32::LANES {
            let mut arr = vec![1; A::u32::LANES];
            A::u32::load_partial(&arr[..lanes]).store(&mut arr);
            assert_eq!(arr.iter().sum::<u32>() as usize, lanes);
        }

        for lanes in 0..A::u32::LANES {
            let mut arr = vec![0; A::u32::LANES];
            A::u32::from(1).store_partial(&mut arr[..lanes]);
            assert_eq!(arr.iter().sum::<u32>() as usize, lanes);
        }
    }

    #[test]
    fn scalar_u32() {
        test_u32::<Scalar>();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(target_feature = "sse2")]
    #[test]
    fn sse2_u32() {
        test_u32::<Sse2>();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[cfg(target_feature = "avx2")]
    #[test]
    fn avx2_u32() {
        test_u32::<Avx2>();
    }

    #[cfg(target_arch = "aarch64")]
    #[cfg(target_feature = "neon")]
    #[test]
    fn neon_u32() {
        test_u32::<Neon>();
    }
}
