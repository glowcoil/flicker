use std::mem;

use crate::simd::*;
use crate::{geom::Vec2, Color};

#[derive(Copy, Clone)]
pub struct Segment {
    pub p1: Vec2,
    pub p2: Vec2,
}

struct Consts<A: Arch>(std::marker::PhantomData<A>);

impl<A: Arch> Consts<A> {
    const BITS_PER_BITMASK: usize = u64::BITS as usize;

    const PIXELS_PER_BIT: usize = A::u32::LANES;
    const PIXELS_PER_BIT_SHIFT: usize = Self::PIXELS_PER_BIT.trailing_zeros() as usize;

    const PIXELS_PER_BITMASK: usize = Self::PIXELS_PER_BIT * Self::BITS_PER_BITMASK;
    const PIXELS_PER_BITMASK_SHIFT: usize = Self::PIXELS_PER_BITMASK.trailing_zeros() as usize;
}

struct Methods {
    bitmask_count_for_width: fn(width: usize) -> usize,
    add_segments: fn(&mut Rasterizer, segments: &[Segment]),
    finish: fn(&mut Rasterizer, color: Color, data: &mut [u32], stride: usize),
}

impl Methods {
    fn specialize() -> Methods {
        struct Specialize;

        impl WithArch for Specialize {
            type Result = Methods;

            fn run<A: Arch>(self) -> Methods {
                Methods {
                    bitmask_count_for_width: bitmask_count_for_width::<A>,
                    add_segments: Rasterizer::add_segments_inner::<A>,
                    finish: Rasterizer::finish_inner::<A>,
                }
            }
        }

        if let Some(methods) = Avx2::try_with(Specialize) {
            methods
        } else {
            Sse2::with(Specialize)
        }
    }
}

pub struct Rasterizer {
    methods: Methods,
    width: usize,
    height: usize,
    coverage: Vec<f32>,
    bitmasks_width: usize,
    bitmasks: Vec<u64>,
}

/// Round up to integer number of bitmasks.
fn bitmask_count_for_width<A: Arch>(width: usize) -> usize {
    (width + Consts::<A>::PIXELS_PER_BITMASK - 1) >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT
}

impl Rasterizer {
    pub fn with_size(width: usize, height: usize) -> Rasterizer {
        let methods = Methods::specialize();

        let bitmasks_width = (methods.bitmask_count_for_width)(width);

        Rasterizer {
            methods,
            width,
            height,
            coverage: vec![0.0; width * height],
            bitmasks_width,
            bitmasks: vec![0; bitmasks_width * height],
        }
    }

    pub fn set_size(&mut self, width: usize, height: usize) {
        self.width = width;
        self.bitmasks_width = (self.methods.bitmask_count_for_width)(width);
        self.height = height;
    }

    pub fn add_segments(&mut self, segments: &[Segment]) {
        (self.methods.add_segments)(self, segments)
    }

    fn add_segments_inner<A: Arch>(&mut self, segments: &[Segment]) {
        invoke!(A, {
            for Segment { p1, p2 } in segments {
                if p1.y == p2.y {
                    continue;
                }

                let y_min = p1.y.min(p2.y);
                let y_max = p1.y.max(p2.y);

                let row_min = (y_min as isize).max(0).min(self.height as isize) as usize;
                let row_max = (y_max as isize + 1).max(0).min(self.height as isize) as usize;

                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let dxdy = dx / dy;
                let dydx = (dy / dx).copysign(dy);

                for row in row_min..row_max {
                    let row_y = row as f32;
                    let row_y1 = row_y.max(y_min);
                    let row_y2 = (row_y + 1.0).min(y_max);

                    let row_y1_x = p1.x + dxdy * (row_y1 - p1.y);
                    let row_y2_x = p1.x + dxdy * (row_y2 - p1.y);

                    let row_x1 = row_y1_x.min(row_y2_x);
                    let row_x2 = row_y1_x.max(row_y2_x);

                    let row_x1_int = row_x1 as isize;
                    let row_x2_int = row_x2 as isize;

                    let col_min = row_x1_int.max(0).min(self.width as isize) as usize;
                    let col_max = (row_x2_int + 1).max(0).min(self.width as isize) as usize;

                    let row_start = row * self.width;
                    let mut carry = 0.0;

                    if row_x1 < 0.0 {
                        carry = if row_x2 < 0.0 {
                            (row_y2 - row_y1).copysign(dy)
                        } else {
                            dydx * (0.0 - row_x1)
                        };
                    }

                    if row_x1_int == row_x2_int {
                        let col = col_min;
                        if col < col_max {
                            let col_x = col as f32;

                            let height = (row_y2 - row_y1).copysign(dy);
                            let area = 0.5 * height * ((col_x + 1.0 - row_x1) + (col_x + 1.0 - row_x2));

                            self.coverage[row_start + col] += carry + area;
                            carry = height - area;
                        }
                    } else {
                        let mut col = col_min;
                        if col < col_max {
                            let col_x = col as f32;

                            let height = dydx * (col_x + 1.0 - row_x1);
                            let area = 0.5 * height * (1.0 + col_x - row_x1);

                            self.coverage[row_start + col] += carry + area;
                            carry = height - area;

                            col += 1;
                        }

                        let area = 0.5 * dydx;
                        while col + 1 < col_max {
                            self.coverage[row_start + col] += carry + area;
                            carry = area;
                            col += 1;
                        }

                        if col < col_max {
                            let col_x = col as f32;

                            let height = dydx * (row_x2 - col_x);
                            let area = 0.5 * height * (2.0 + col_x - row_x2);

                            self.coverage[row_start + col] += carry + area;
                            carry = height - area;

                            col += 1;
                        }
                    }

                    if col_max < self.width {
                        self.coverage[row_start + col_max] += carry;
                    }

                    for col in col_min..(col_max + 1).min(self.width) {
                        let bitmask_index = row * self.bitmasks_width
                            + (col >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT);
                        let bit_index = (col >> Consts::<A>::PIXELS_PER_BIT_SHIFT)
                            & (Consts::<A>::BITS_PER_BITMASK - 1);
                        self.bitmasks[bitmask_index] |= 1 << bit_index;
                    }
                }
            }
        })
    }

    pub fn finish(&mut self, color: Color, data: &mut [u32], stride: usize) {
        (self.methods.finish)(self, color, data, stride)
    }

    fn finish_inner<A: Arch>(&mut self, color: Color, data: &mut [u32], stride: usize) {
        invoke!(A, {
            let a_unit = A::f32::from(color.a() as f32 * (1.0 / 255.0));
            let src = Pixels {
                a: A::f32::from(color.a() as f32),
                r: a_unit * A::f32::from(color.r() as f32),
                g: a_unit * A::f32::from(color.g() as f32),
                b: a_unit * A::f32::from(color.b() as f32),
            };

            for y in 0..self.height {
                let mut accum = 0.0;
                let mut coverage = 0.0;

                let coverage_start = y * self.width;
                let coverage_end = coverage_start + self.width;
                let coverage_row = &mut self.coverage[coverage_start..coverage_end];

                let pixels_start = y * stride;
                let pixels_end = pixels_start + self.width;
                let pixels_row = &mut data[pixels_start..pixels_end];

                let bitmasks_start = y * self.bitmasks_width;
                let bitmasks_end = bitmasks_start + self.bitmasks_width;
                let bitmasks_row = &mut self.bitmasks[bitmasks_start..bitmasks_end];

                let mut x = 0;
                let mut bitmask_index = 0;
                let mut bitmask = mem::replace(&mut bitmasks_row[0], 0);
                loop {
                    // Find next 1 bit (or the end of the scanline).
                    let next_x;
                    loop {
                        if bitmask != 0 {
                            let offset = bitmask.trailing_zeros() as usize;
                            bitmask |= !(!0 << offset);
                            let bitmask_base =
                                bitmask_index << Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                            next_x = (bitmask_base + (offset << Consts::<A>::PIXELS_PER_BIT_SHIFT))
                                .min(self.width);
                            break;
                        }

                        bitmask_index += 1;
                        if bitmask_index == self.bitmasks_width {
                            next_x = self.width;
                            break;
                        }

                        bitmask = mem::replace(&mut bitmasks_row[bitmask_index], 0);
                    }

                    // Composite an interior span (or skip an empty span).
                    if next_x > x {
                        if coverage > 254.5 / 255.0 && color.a() == 255 {
                            pixels_row[x..next_x].fill(color.into());
                        } else if coverage > 0.5 / 255.0 {
                            let mut pixels_chunks =
                                pixels_row[x..next_x].chunks_exact_mut(A::u32::LANES);

                            for pixels_slice in &mut pixels_chunks {
                                let mask = A::f32::from(coverage);
                                let dst = Pixels::<A>::unpack(A::u32::load(pixels_slice));
                                dst.blend(src, mask).pack().store(pixels_slice);
                            }

                            let pixels_remainder = pixels_chunks.into_remainder();
                            if !pixels_remainder.is_empty() {
                                let mask = A::f32::from(coverage);
                                let dst = Pixels::unpack(A::u32::load_partial(pixels_remainder));
                                dst.blend(src, mask).pack().store_partial(pixels_remainder);
                            }
                        }
                    }

                    x = next_x;
                    if next_x == self.width {
                        break;
                    }

                    // Find next 0 bit (or the end of the scanline).
                    let next_x;
                    loop {
                        if bitmask != !0 {
                            let offset = bitmask.trailing_ones() as usize;
                            bitmask &= !0 << offset;
                            let bitmask_base =
                                bitmask_index << Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                            next_x = (bitmask_base + (offset << Consts::<A>::PIXELS_PER_BIT_SHIFT))
                                .min(self.width);
                            break;
                        }

                        bitmask_index += 1;
                        if bitmask_index == self.bitmasks_width {
                            next_x = self.width;
                            break;
                        }

                        bitmask = mem::replace(&mut bitmasks_row[bitmask_index], 0);
                    }

                    // Composite an edge span.
                    if next_x > x {
                        let coverage_slice = &mut coverage_row[x..next_x];
                        let mut coverage_chunks = coverage_slice.chunks_exact_mut(A::f32::LANES);

                        let pixels_slice = &mut pixels_row[x..next_x];
                        let mut pixels_chunks = pixels_slice.chunks_exact_mut(A::u32::LANES);

                        for (coverage_chunk, pixels_chunk) in
                            (&mut coverage_chunks).zip(&mut pixels_chunks)
                        {
                            let deltas = A::f32::load(coverage_chunk);
                            let accums = A::f32::from(accum) + deltas.scan_sum();
                            accum = accums.last();
                            let mask = accums.abs().min(A::f32::from(1.0));
                            coverage = mask.last();

                            coverage_chunk.fill(0.0);

                            let dst = Pixels::unpack(A::u32::load(pixels_chunk));
                            dst.blend(src, mask).pack().store(pixels_chunk);
                        }

                        let coverage_remainder = coverage_chunks.into_remainder();
                        let pixels_remainder = pixels_chunks.into_remainder();
                        if !pixels_remainder.is_empty() && !coverage_remainder.is_empty() {
                            let deltas = A::f32::load_partial(coverage_remainder);
                            let accums = A::f32::from(accum) + deltas.scan_sum();
                            accum = accums.last();
                            let mask = accums.abs().min(A::f32::from(1.0));
                            coverage = mask.last();

                            coverage_remainder.fill(0.0);

                            let dst = Pixels::unpack(A::u32::load_partial(pixels_remainder));
                            dst.blend(src, mask).pack().store_partial(pixels_remainder);
                        }
                    }

                    x = next_x;
                    if next_x == self.width {
                        break;
                    }
                }
            }
        })
    }
}

struct Pixels<A: Arch> {
    a: A::f32,
    r: A::f32,
    g: A::f32,
    b: A::f32,
}

impl<A: Arch> Clone for Pixels<A> {
    fn clone(&self) -> Self {
        Pixels {
            a: self.a,
            r: self.r,
            g: self.g,
            b: self.b,
        }
    }
}

impl<A: Arch> Copy for Pixels<A> {}

impl<A: Arch> Pixels<A> {
    #[inline]
    fn unpack(data: A::u32) -> Self {
        invoke!(A, {
            Pixels {
                a: A::f32::from((data >> 24) & A::u32::from(0xFF)),
                r: A::f32::from((data >> 16) & A::u32::from(0xFF)),
                g: A::f32::from((data >> 8) & A::u32::from(0xFF)),
                b: A::f32::from((data >> 0) & A::u32::from(0xFF)),
            }
        })
    }

    #[inline]
    fn pack(self) -> A::u32 {
        invoke!(A, {
            let a = A::u32::from(self.a);
            let r = A::u32::from(self.r);
            let g = A::u32::from(self.g);
            let b = A::u32::from(self.b);

            (a << 24) | (r << 16) | (g << 8) | (b << 0)
        })
    }

    #[inline]
    fn blend(self, src: Self, mask: A::f32) -> Self {
        invoke!(A, {
            let inv_a = A::f32::from(1.0) - mask * A::f32::from(1.0 / 255.0) * src.a;
            Pixels {
                a: mask * src.a + inv_a * self.a,
                r: mask * src.r + inv_a * self.r,
                g: mask * src.g + inv_a * self.g,
                b: mask * src.b + inv_a * self.b,
            }
        })
    }
}
