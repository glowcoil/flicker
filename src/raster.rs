use std::mem;

use crate::simd::*;
use crate::{geom::Vec2, Color};

#[derive(Copy, Clone)]
pub struct Segment {
    pub p1: Vec2,
    pub p2: Vec2,
}

const BITS_PER_BITMASK: usize = u64::BITS as usize;
const BITS_PER_BITMASK_SHIFT: usize = BITS_PER_BITMASK.trailing_zeros() as usize;

const PIXELS_PER_BIT: usize = 4;
const PIXELS_PER_BIT_SHIFT: usize = PIXELS_PER_BIT.trailing_zeros() as usize;

const PIXELS_PER_BITMASK: usize = PIXELS_PER_BIT * BITS_PER_BITMASK;
const PIXELS_PER_BITMASK_SHIFT: usize = PIXELS_PER_BITMASK.trailing_zeros() as usize;

trait FlipCoords {
    fn winding(value: f32) -> f32;
    fn row(y: usize, height: usize) -> usize;
    fn y_coord(p: Vec2, height: f32) -> Vec2;
}

struct PosXPosY;

impl FlipCoords for PosXPosY {
    #[inline(always)]
    fn winding(value: f32) -> f32 {
        value
    }

    #[inline(always)]
    fn row(y: usize, _height: usize) -> usize {
        y
    }

    #[inline(always)]
    fn y_coord(p: Vec2, _height: f32) -> Vec2 {
        Vec2::new(p.x, p.y)
    }
}

struct PosXNegY;

impl FlipCoords for PosXNegY {
    #[inline(always)]
    fn winding(value: f32) -> f32 {
        -value
    }

    #[inline(always)]
    fn row(y: usize, height: usize) -> usize {
        height - 1 - y
    }

    #[inline(always)]
    fn y_coord(p: Vec2, height: f32) -> Vec2 {
        Vec2::new(p.x, height - p.y)
    }
}

struct NegXPosY;

impl FlipCoords for NegXPosY {
    #[inline(always)]
    fn winding(value: f32) -> f32 {
        value
    }

    #[inline(always)]
    fn row(y: usize, height: usize) -> usize {
        height - 1 - y
    }

    #[inline(always)]
    fn y_coord(p: Vec2, height: f32) -> Vec2 {
        Vec2::new(p.x, height - p.y)
    }
}

struct NegXNegY;

impl FlipCoords for NegXNegY {
    #[inline(always)]
    fn winding(value: f32) -> f32 {
        -value
    }

    #[inline(always)]
    fn row(y: usize, _height: usize) -> usize {
        y
    }

    #[inline(always)]
    fn y_coord(p: Vec2, _height: f32) -> Vec2 {
        Vec2::new(p.x, p.y)
    }
}

pub struct Rasterizer {
    width: usize,
    height: usize,
    coverage: Vec<f32>,
    bitmasks_width: usize,
    bitmasks: Vec<u64>,
}

/// Round up to integer number of bitmasks.
fn bitmask_count_for_width(width: usize) -> usize {
    (width + PIXELS_PER_BITMASK - 1) >> PIXELS_PER_BITMASK_SHIFT
}

// On baseline x86_64, f32::floor gets lowered to a function call, so this is significantly faster.
#[inline]
fn floor(x: f32) -> i32 {
    let mut result = x as i32;
    if x < 0.0 {
        result -= 1;
    }
    result
}

impl Rasterizer {
    pub fn new() -> Rasterizer {
        Rasterizer {
            width: 0,
            height: 0,
            coverage: Vec::new(),
            bitmasks_width: 0,
            bitmasks: Vec::new(),
        }
    }

    pub fn set_size(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;

        let coverage_size = self.width * self.height;
        if self.coverage.len() < coverage_size {
            self.coverage.resize(coverage_size, 0.0);
        }

        self.bitmasks_width = bitmask_count_for_width(self.width);

        let bitmasks_size = self.bitmasks_width * self.height;
        if self.bitmasks.len() < bitmasks_size {
            self.bitmasks.resize(bitmasks_size, 0);
        }
    }

    pub fn add_segments(&mut self, segments: &[Segment]) {
        for segment in segments {
            if segment.p1.x < segment.p2.x {
                if segment.p1.y < segment.p2.y {
                    self.add_segment::<PosXPosY>(segment.p1, segment.p2);
                } else {
                    self.add_segment::<PosXNegY>(segment.p1, segment.p2);
                }
            } else {
                if segment.p1.y < segment.p2.y {
                    self.add_segment::<NegXPosY>(segment.p2, segment.p1);
                } else {
                    self.add_segment::<NegXNegY>(segment.p2, segment.p1);
                }
            }
        }
    }

    #[inline(always)]
    fn add_segment<Flip: FlipCoords>(&mut self, p1: Vec2, p2: Vec2) {
        let p1 = Flip::y_coord(p1, self.height as f32);
        let p2 = Flip::y_coord(p2, self.height as f32);

        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let dxdy = dx / dy;
        let dydx = dy / dx;

        let mut y = floor(p1.y);
        let mut y_offset = p1.y - y as f32;

        let mut y_end = floor(p2.y);
        let mut y_offset_end = p2.y - y_end as f32;

        let mut x = floor(p1.x);
        let mut x_offset = p1.x - x as f32;

        let mut x_end = floor(p2.x);
        let mut x_offset_end = p2.x - x_end as f32;

        if y >= self.height as i32 {
            return;
        }

        if y_end < 0 {
            return;
        }

        if y < 0 {
            let clip_x = p1.x - dxdy * p1.y;
            x = floor(clip_x);
            x_offset = clip_x - x as f32;

            y = 0;
            y_offset = 0.0;
        }

        if y_end >= self.height as i32 {
            let clip_x = p1.x + dxdy * (self.height as f32 - p1.y);
            x_end = floor(clip_x);
            x_offset_end = clip_x - x as f32;

            y_end = self.height as i32 - 1;
            y_offset_end = 1.0;
        }

        if x >= self.width as i32 {
            return;
        }

        if x < 0 {
            let mut y_split = y_end;
            let mut y_offset_split = y_offset_end;
            if x_end >= 0 {
                let y_clip = p1.y - dydx * p1.x;
                y_split = floor(y_clip).min(self.height as i32 - 1);
                y_offset_split = y_clip - y_split as f32;
            }

            while y < y_split {
                let row = Flip::row(y as usize, self.height);
                self.coverage[row * self.width] += Flip::winding(1.0 - y_offset);
                self.bitmasks[row * self.bitmasks_width] |= 1;

                y += 1;
                y_offset = 0.0;
            }

            let row = Flip::row(y as usize, self.height);
            self.coverage[row * self.width] += Flip::winding(y_offset_split - y_offset);
            self.bitmasks[row * self.bitmasks_width] |= 1;

            x = 0;
            x_offset = 0.0;
            y_offset = y_offset_split;
        }

        if x_end < 0 {
            return;
        }

        if x_end >= self.width as i32 {
            x_end = self.width as i32 - 1;
            x_offset_end = 1.0;

            let clip_y = p2.y - dydx * (p2.x - self.width as f32);
            y_end = floor(clip_y);
            y_offset_end = clip_y - y_end as f32;
        }

        let mut x_offset_next = x_offset + dxdy * (1.0 - y_offset);
        let mut y_offset_next = y_offset + dydx * (1.0 - x_offset);

        while y < y_end {
            let row = Flip::row(y as usize, self.height);
            let row_start = x as usize;
            while y_offset_next < 1.0 {
                let height = Flip::winding(y_offset_next - y_offset);
                let area = 0.5 * height * (1.0 - x_offset);

                self.coverage[row * self.width + x as usize] += area;
                self.coverage[row * self.width + x as usize + 1] += height - area;

                x += 1;
                x_offset = 0.0;
                x_offset_next -= 1.0;

                y_offset = y_offset_next;
                y_offset_next += dydx;
            }

            let height = Flip::winding(1.0 - y_offset);
            let area = 0.5 * height * (2.0 - x_offset - x_offset_next);

            let mut row_end = x as usize + 1;
            self.coverage[row * self.width + x as usize] += area;
            if x as usize + 1 < self.width {
                self.coverage[row * self.width + x as usize + 1] += height - area;
                row_end += 1;
            }
            self.fill_cells(row, row_start, row_end);

            x_offset = x_offset_next;
            x_offset_next += dxdy;

            y += 1;
            y_offset = 0.0;
            y_offset_next -= 1.0;
        }

        let row = Flip::row(y as usize, self.height);
        let row_start = x as usize;
        while x < x_end {
            let height = Flip::winding(y_offset_next - y_offset);
            let area = 0.5 * height * (1.0 - x_offset);

            self.coverage[row * self.width + x as usize] += area;
            self.coverage[row * self.width + x as usize + 1] += height - area;

            x += 1;
            x_offset = 0.0;
            x_offset_next -= 1.0;

            y_offset = y_offset_next;
            y_offset_next += dydx;
        }

        let height = Flip::winding(y_offset_end - y_offset);
        let area = 0.5 * height * (2.0 - x_offset - x_offset_end);

        let mut row_end = x as usize + 1;
        self.coverage[row * self.width + x as usize] += area;
        if x as usize + 1 < self.width {
            self.coverage[row * self.width + x as usize + 1] += height - area;
            row_end += 1;
        }
        self.fill_cells(row, row_start, row_end);
    }

    #[inline]
    fn fill_cells(&mut self, y: usize, start: usize, end: usize) {
        let offset = y * self.bitmasks_width;

        let cell_min = start >> PIXELS_PER_BIT_SHIFT;
        let cell_max = (end + PIXELS_PER_BIT - 1) >> PIXELS_PER_BIT_SHIFT;
        let bitmask_index_min = cell_min >> BITS_PER_BITMASK_SHIFT;
        let bitmask_index_max = cell_max >> BITS_PER_BITMASK_SHIFT;

        let mut bit_min = cell_min & (BITS_PER_BITMASK - 1);
        for bitmask_index in bitmask_index_min..bitmask_index_max {
            self.bitmasks[offset + bitmask_index] |= !0 << bit_min;
            bit_min = 0;
        }
        let bit_max = cell_max & (BITS_PER_BITMASK - 1);
        self.bitmasks[offset + bitmask_index_max] |= (!0 << bit_min) & !(!0 << bit_max);
    }

    pub fn finish(&mut self, color: Color, data: &mut [u32], stride: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx2")]
            return self.finish_inner::<Avx2>(color, data, stride);

            #[cfg(all(not(target_feature = "avx2"), target_feature = "sse2"))]
            return self.finish_inner::<Sse2>(color, data, stride);

            #[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
            return self.finish_inner::<Scalar>(color, data, stride);
        }

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            return self.finish_inner::<Neon>(color, data, stride);
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        self.finish_inner::<Scalar>(color, data, stride)
    }

    fn finish_inner<A: Arch>(&mut self, color: Color, data: &mut [u32], stride: usize) {
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
                        let bitmask_base = bitmask_index << PIXELS_PER_BITMASK_SHIFT;
                        next_x = (bitmask_base + (offset << PIXELS_PER_BIT_SHIFT)).min(self.width);
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
                        let bitmask_base = bitmask_index << PIXELS_PER_BITMASK_SHIFT;
                        next_x = (bitmask_base + (offset << PIXELS_PER_BIT_SHIFT)).min(self.width);
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
        }
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
        Pixels {
            a: A::f32::from((data >> 24) & A::u32::from(0xFF)),
            r: A::f32::from((data >> 16) & A::u32::from(0xFF)),
            g: A::f32::from((data >> 8) & A::u32::from(0xFF)),
            b: A::f32::from((data >> 0) & A::u32::from(0xFF)),
        }
    }

    #[inline]
    fn pack(self) -> A::u32 {
        let a = A::u32::from(self.a);
        let r = A::u32::from(self.r);
        let g = A::u32::from(self.g);
        let b = A::u32::from(self.b);

        (a << 24) | (r << 16) | (g << 8) | (b << 0)
    }

    #[inline]
    fn blend(self, src: Self, mask: A::f32) -> Self {
        let inv_a = A::f32::from(1.0) - mask * A::f32::from(1.0 / 255.0) * src.a;
        Pixels {
            a: mask * src.a + inv_a * self.a,
            r: mask * src.r + inv_a * self.r,
            g: mask * src.g + inv_a * self.g,
            b: mask * src.b + inv_a * self.b,
        }
    }
}
