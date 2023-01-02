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

#[cfg(test)]
mod tests {
    use super::*;

    struct TestF32;

    impl Task for TestF32 {
        type Result = ();

        #[inline(always)]
        fn run<A: Arch>(self) {
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
                    let result = A::f32::from_slice(chunk_a) + A::f32::from_slice(chunk_b);
                    for ((&a, &b), &c) in chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .zip(result.as_slice().iter())
                    {
                        let correct = a + b;
                        assert!(
                            correct.to_bits() == c.to_bits(),
                            "expected {a} + {b} == {correct}, got {c}"
                        );
                    }

                    let result = A::f32::from_slice(chunk_a) - A::f32::from_slice(chunk_b);
                    for ((&a, &b), &c) in chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .zip(result.as_slice().iter())
                    {
                        let correct = a - b;
                        assert!(
                            correct.to_bits() == c.to_bits(),
                            "expected {a} - {b} == {correct}, got {c}"
                        );
                    }

                    let result = A::f32::from_slice(chunk_a) * A::f32::from_slice(chunk_b);
                    for ((&a, &b), &c) in chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .zip(result.as_slice().iter())
                    {
                        let correct = a * b;
                        assert!(
                            correct.to_bits() == c.to_bits(),
                            "expected {a} * {b} == {correct}, got {c}"
                        );
                    }

                    let result = A::f32::from_slice(chunk_a) / A::f32::from_slice(chunk_b);
                    for ((&a, &b), &c) in chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .zip(result.as_slice().iter())
                    {
                        let correct = a / b;
                        assert!(
                            correct.to_bits() == c.to_bits(),
                            "expected {a} / {b} == {correct}, got {c}"
                        );
                    }
                }
            }

            for chunk in values.chunks(A::f32::LANES) {
                let result = -A::f32::from_slice(chunk);
                for (&a, &b) in chunk.iter().zip(result.as_slice().iter()) {
                    assert!((-a).to_bits() == b.to_bits(), "-{a} == {b}");
                }
            }

            // Test for scan_sum. Since different backends use different reduction trees, we can't
            // expect bit equality in the face of rounding or edge cases like NaN or infinity, so
            // just test on exactly representable integers for now.

            let values = (0..64).map(|x| x as f32).collect::<Vec<f32>>();

            for chunk in values.chunks(A::f32::LANES) {
                let result = A::f32::from_slice(chunk).scan_sum();
                let mut correct = A::f32::from_slice(chunk);
                let mut accum = 0.0;
                for x in correct.as_mut_slice() {
                    accum += *x;
                    *x = accum;
                }
                let equal = result
                    .as_slice()
                    .iter()
                    .zip(correct.as_slice().iter())
                    .all(|(a, b)| a == b);
                assert!(equal, "scan_sum() failed\n   input: {chunk:?}\nexpected: {correct:?}\n     got: {result:?}");
            }
        }
    }

    #[test]
    fn scalar_f32() {
        let test = Scalar::specialize::<TestF32>();
        test(TestF32);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_f32() {
        if let Some(test) = Sse2::try_specialize::<TestF32>() {
            test(TestF32);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_f32() {
        if let Some(test) = Avx2::try_specialize::<TestF32>() {
            test(TestF32);
        }
    }
}
