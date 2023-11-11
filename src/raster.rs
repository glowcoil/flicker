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
    const BITS_PER_BITMASK_SHIFT: usize = Self::BITS_PER_BITMASK.trailing_zeros() as usize;

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

        #[cfg(target_arch = "x86_64")]
        if let Some(methods) = Avx2::try_with(Specialize) {
            methods
        } else {
            Sse2::with(Specialize)
        }

        #[cfg(target_arch = "x86")]
        if let Some(methods) = Avx2::try_with(Specialize) {
            methods
        } else if let Some(methods) = Sse2::try_with(specialize) {
            methods
        } else {
            Scalar::with(Specialize)
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(methods) = Neon::try_with(Specialize) {
            methods
        } else {
            Scalar::with(Specialize)
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        Scalar::with(Specialize)
    }
}

pub struct Rasterizer {
    methods: Methods,
    width: usize,
    stride: usize,
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

        let stride = width + 1;
        let bitmasks_width = (methods.bitmask_count_for_width)(stride);

        Rasterizer {
            methods,
            width,
            stride,
            height,
            coverage: vec![0.0; stride * height],
            bitmasks_width,
            bitmasks: vec![0; bitmasks_width * height],
        }
    }

    pub fn set_size(&mut self, width: usize, height: usize) {
        self.width = width;
        self.stride = width + 1;
        self.bitmasks_width = (self.methods.bitmask_count_for_width)(self.stride);
        self.height = height;
    }

    pub fn add_segments(&mut self, segments: &[Segment]) {
        (self.methods.add_segments)(self, segments)
    }

    fn add_segments_inner<A: Arch>(&mut self, segments: &[Segment]) {
        invoke!(A, {
            for segment in segments {
                self.add_segment::<A>(segment.p1, segment.p2);
            }
        })
    }

    fn add_segment<A: Arch>(&mut self, p1: Vec2, p2: Vec2) {
        invoke!(A, {
            let p_left;
            let p_right;
            if p1.x < p2.x {
                p_left = p1;
                p_right = p2;
            } else {
                p_left = p2;
                p_right = p1;
            }

            let sign = (p2.y - p1.y).signum();

            let mut x = p_left.x.floor() as isize;
            let mut y = p_left.y.floor() as isize;

            let x_end = p_right.x.floor() as isize;
            let y_end = p_right.y.floor() as isize;

            let dx = p_right.x - p_left.x;
            let mut x_offset = p_left.x - x as f32;
            let x_offset_end = p_right.x - x_end as f32;

            let y_inc;
            let mut y_offset;
            let y_offset_end;
            let dy;
            let dxdy;
            let dydx;
            let mut x_offset_next;
            let mut y_offset_next;
            let y_out;
            if p_left.y < p_right.y {
                y_inc = 1;
                y_offset = p_left.y - y as f32;
                y_offset_end = p_right.y - y_end as f32;
                dy = p_right.y - p_left.y;

                if y >= self.height as isize {
                    return;
                }

                if y_end < 0 {
                    return;
                }

                dxdy = dx / dy;

                x_offset_next = x_offset + dxdy * (1.0 - y_offset);
                if dx == 0.0 {
                    dydx = 0.0;
                    y_offset_next = f32::INFINITY;
                } else {
                    dydx = dy / dx;
                    y_offset_next = y_offset + dydx * (1.0 - x_offset);
                };

                if y < 0 {
                    let x_jump = x_offset_next + dxdy * (-y - 1) as f32;
                    let x_inc = x_jump as isize;
                    x += x_inc;
                    x_offset = x_jump - x_inc as f32;
                    x_offset_next = x_offset + dxdy;

                    y_offset_next += dydx * x_inc as f32 + y as f32;
                    y = 0;
                    y_offset = 0.0;
                }

                y_out = self.height as isize;
            } else {
                y_inc = -1;
                y_offset = 1.0 - (p_left.y - y as f32);
                y_offset_end = 1.0 - (p_right.y - y_end as f32);
                dy = p_left.y - p_right.y;

                if y < 0 {
                    return;
                }

                if y_end >= self.height as isize {
                    return;
                }

                dxdy = dx / dy;

                x_offset_next = x_offset + dxdy * (1.0 - y_offset);
                if dx == 0.0 {
                    dydx = 0.0;
                    y_offset_next = f32::INFINITY;
                } else {
                    dydx = dy / dx;
                    y_offset_next = y_offset + dydx * (1.0 - x_offset);
                };

                if y >= self.height as isize {
                    let x_jump = x_offset_next + dxdy * (y - self.height as isize) as f32;
                    let x_inc = x_jump as isize;
                    x += x_inc;
                    x_offset = x_jump - x_inc as f32;
                    x_offset_next = x_offset + dxdy;

                    y_offset_next += dydx * x_inc as f32 - (y - (self.height - 1) as isize) as f32;
                    y = self.height as isize - 1;
                    y_offset = 0.0;
                }

                y_out = -1;
            }

            if x >= self.width as isize {
                return;
            }

            while x < 0 {
                if y == y_end && x == x_end {
                    self.coverage[y as usize * self.stride] += sign * (y_offset_end - y_offset);
                    self.bitmasks[y as usize * self.bitmasks_width] |= 1;

                    return;
                }

                if y_offset_next > 1.0 {
                    self.coverage[y as usize * self.stride] += sign * (1.0 - y_offset);
                    self.bitmasks[y as usize * self.bitmasks_width] |= 1;

                    x_offset = x_offset_next;
                    x_offset_next += dxdy;

                    y += y_inc;
                    if y == y_out {
                        return;
                    }

                    y_offset = 0.0;
                    y_offset_next -= 1.0;

                    continue;
                }

                self.coverage[y as usize * self.stride] += sign * (y_offset_next - y_offset);
                self.bitmasks[y as usize * self.bitmasks_width] |= 1;

                x += 1;

                x_offset = 0.0;
                x_offset_next -= 1.0;

                y_offset = y_offset_next;
                y_offset_next += dydx;
            }

            let mut row_start = x as usize;
            let mut carry = 0.0;
            loop {
                if y == y_end && x == x_end {
                    let height = sign * (y_offset_end - y_offset);
                    let area = 0.5 * height * (2.0 - x_offset - x_offset_end);
                    self.coverage[y as usize * self.stride + x_end as usize] += carry + area;
                    self.coverage[y as usize * self.stride + x_end as usize + 1] += height - area;
                    self.fill_cells::<A>(y_end as usize, row_start, x_end as usize + 2);

                    return;
                }

                if y_offset_next > 1.0 {
                    let height = sign * (1.0 - y_offset);
                    let area = 0.5 * height * (2.0 - x_offset - x_offset_next);
                    self.coverage[y as usize * self.stride + x as usize] += carry + area;
                    self.coverage[y as usize * self.stride + x as usize + 1] += height - area;
                    self.fill_cells::<A>(y as usize, row_start, x as usize + 2);

                    x_offset = x_offset_next;
                    x_offset_next += dxdy;

                    y += y_inc;
                    if y == y_out {
                        return;
                    }

                    y_offset = 0.0;
                    y_offset_next -= 1.0;

                    row_start = x as usize;
                    carry = 0.0;

                    continue;
                }

                let height = sign * (y_offset_next - y_offset);
                let area = 0.5 * height * (1.0 - x_offset);
                self.coverage[y as usize * self.stride + x as usize] += carry + area;
                carry = height - area;

                x += 1;
                if x as usize == self.width {
                    self.fill_cells::<A>(y as usize, row_start, self.width);

                    return;
                }

                x_offset = 0.0;
                x_offset_next -= 1.0;

                y_offset = y_offset_next;
                y_offset_next += dydx;
            }
        })
    }

    #[inline]
    fn fill_cells<A: Arch>(&mut self, y: usize, start: usize, end: usize) {
        let offset = y * self.bitmasks_width;

        let cell_min = start >> Consts::<A>::PIXELS_PER_BIT_SHIFT;
        let cell_max = (end + Consts::<A>::PIXELS_PER_BIT - 1) >> Consts::<A>::PIXELS_PER_BIT_SHIFT;
        let bitmask_index_min = cell_min >> Consts::<A>::BITS_PER_BITMASK_SHIFT;
        let bitmask_index_max = cell_max >> Consts::<A>::BITS_PER_BITMASK_SHIFT;

        let mut bit_min = cell_min & (Consts::<A>::BITS_PER_BITMASK - 1);
        for bitmask_index in bitmask_index_min..bitmask_index_max {
            self.bitmasks[offset + bitmask_index] |= !0 << bit_min;
            bit_min = 0;
        }
        let bit_max = cell_max & (Consts::<A>::BITS_PER_BITMASK - 1);
        self.bitmasks[offset + bitmask_index_max] |= (!0 << bit_min) & !(!0 << bit_max);
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

                let coverage_start = y * self.stride;
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
                            let accums = A::f32::from(accum) + deltas.prefix_sum();
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
                            let accums = A::f32::from(accum) + deltas.prefix_sum();
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

                self.coverage[coverage_end] = 0.0;
                self.bitmasks[bitmasks_end - 1] = 0;
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
