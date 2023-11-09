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
    height: usize,
    stride: usize,
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
            height,
            stride,
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
            for Segment { p1, p2 } in segments {
                let mut x = (p1.x + 1.0) as isize - 1;
                let mut y = (p1.y + 1.0) as isize - 1;

                let x_end = (p2.x + 1.0) as isize - 1;
                let y_end = (p2.y + 1.0) as isize - 1;

                if y == y_end {
                    if y < 0 || y >= self.height as isize {
                        continue;
                    }

                    if x == x_end {
                        if x >= self.width as isize{
                            continue;
                        }

                        let row_offset = y as usize * self.stride;
                        let bitmask_offset = y as usize * self.bitmasks_width;

                        if x < 0 {
                            let height = p2.y - p1.y;
                            self.coverage[row_offset] += height;
                            self.bitmasks[bitmask_offset] |= 1;

                            continue;
                        }

                        let x = x as usize;

                        let height = p2.y - p1.y;
                        let area = 0.5 * height * ((x as f32 + 1.0 - p1.x) + (x as f32 + 1.0 - p2.x));
                        self.coverage[row_offset + x] += area;
                        self.coverage[row_offset + x + 1] += height - area;

                        let bitmask_index_1 = x >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                        let bitmask_index_2 = (x + 1) >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                        let bit_index_1 =
                            (x >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
                        let bit_index_2 =
                            ((x + 1) >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
                        if bitmask_index_1 == bitmask_index_2 {
                            self.bitmasks[bitmask_offset + bitmask_index_1] |= (1 << bit_index_1) | (1 << bit_index_2);
                        } else {
                            self.bitmasks[bitmask_offset + bitmask_index_1] |= 1 << bit_index_1;
                            self.bitmasks[bitmask_offset + bitmask_index_2] |= 1 << bit_index_2;
                        }

                        continue;
                    }

                    let p_left;
                    let p_right;
                    let x_min;
                    let x_max;
                    if x < x_end {
                        p_left = p1;
                        p_right = p2;
                        x_min = x;
                        x_max = x_end;
                    } else {
                        p_left = p2;
                        p_right = p1;
                        x_min = x_end;
                        x_max = x;
                    };

                    if x_min >= self.width as isize {
                        continue;
                    }

                    let row_offset = y as usize * self.stride;
                    let bitmask_offset = y as usize * self.bitmasks_width;

                    if x_max < 0 {
                        let height = p2.y - p1.y;
                        self.coverage[row_offset] += height;
                        self.bitmasks[bitmask_offset] |= 1;

                        continue;
                    }

                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let dydx = dy / dx;

                    let mut carry;

                    let span_start;
                    let bits_start;
                    if x_min < 0 {
                        let width = 0.0 - p_left.x;
                        let height = (dydx * width).copysign(dy);
                        carry = height;

                        span_start = 0;
                        bits_start = 0;
                    } else {
                        let width = 1.0 - p_left.x.fract();
                        let height = (dydx * width).copysign(dy);
                        let area = 0.5 * width * height;
                        self.coverage[row_offset + x_min as usize] += area;

                        carry = height - area;

                        span_start = x_min as usize + 1;
                        bits_start = x_min as usize;
                    }

                    let span_end;
                    let bits_end;
                    if x_max >= self.width as isize {
                        span_end = self.width;
                        bits_end = self.width;
                    } else {
                        span_end = x_max as usize;
                        bits_end = x_max as usize + 2;
                    }

                    let area = 0.5 * dydx.copysign(dy);
                    for px in &mut self.coverage[row_offset + span_start..row_offset + span_end] {
                        *px += carry + area;
                        carry = area;
                    }

                    if x_max < self.width as isize {
                        let width = p_right.x.fract();
                        let height = (dydx * width).copysign(dy);
                        let area = 0.5 * (2.0 - width) * height;
                        self.coverage[row_offset + x_max as usize] += carry + area;
                        self.coverage[row_offset + x_max as usize + 1] += height - area;
                    }

                    let cell_min = bits_start as usize >> Consts::<A>::PIXELS_PER_BIT_SHIFT;
                    let cell_max = (bits_end + 2) as usize >> Consts::<A>::PIXELS_PER_BIT_SHIFT;
                    let bitmask_index_min = cell_min >> Consts::<A>::BITS_PER_BITMASK_SHIFT;
                    let bitmask_index_max = cell_max >> Consts::<A>::BITS_PER_BITMASK_SHIFT;
                    
                    if bitmask_index_min == bitmask_index_max {
                        let bit_min = cell_min & (Consts::<A>::BITS_PER_BITMASK - 1);
                        let bit_max = cell_max & (Consts::<A>::BITS_PER_BITMASK - 1);
                        self.bitmasks[bitmask_offset + bitmask_index_min] |= (!0 << bit_min) & !(!0 << bit_max);
                    } else {
                        for cell in cell_min..cell_max {
                            let bitmask_index =
                                bitmask_offset + (cell >> Consts::<A>::BITS_PER_BITMASK_SHIFT);
                            let bit_index = cell & (Consts::<A>::BITS_PER_BITMASK - 1);
                            self.bitmasks[bitmask_index] |= 1 << bit_index;
                        }
                    }

                    continue;
                }

                if x < 0 && x_end < 0 && y != y_end {
                    let p_top;
                    let p_bottom;
                    let y_min;
                    let y_max;
                    if y < y_end {
                        p_top = p1;
                        p_bottom = p2;
                        y_min = y;
                        y_max = y_end;
                    } else {
                        p_top = p2;
                        p_bottom = p1;
                        y_min = y_end;
                        y_max = y;
                    };

                    if y_max < 0 || y_min >= self.height as isize {
                        continue;
                    }

                    let dy = p2.y - p1.y;

                    let span_start;
                    let bits_start;
                    if y_min < 0 {
                        span_start = 0;
                        bits_start = 0;
                    } else {
                        let height = (1.0 - p_top.y.fract()).copysign(dy);
                        self.coverage[y_min as usize * self.stride] += height;

                        span_start = y_min as usize + 1;
                        bits_start = y_min as usize;
                    }

                    let span_end;
                    let bits_end;
                    if y_max >= self.height as isize {
                        span_end = self.height;
                        bits_end = self.height;
                    } else {
                        span_end = y_max as usize;
                        bits_end = y_max as usize + 1;
                    }

                    for row in span_start..span_end {
                        self.coverage[row * self.stride] += 1.0f32.copysign(dy);
                    }

                    if y_max < self.height as isize {
                        let height = p_bottom.y.fract().copysign(dy);
                        self.coverage[y_max as usize * self.stride] += height;
                    }

                    for row in bits_start..bits_end {
                        self.bitmasks[row * self.bitmasks_width] |= 1;
                    }

                    continue;
                }

                if x == x_end {
                    if x >= self.width as isize {
                        continue;
                    }

                    let p_top;
                    let p_bottom;
                    let y_min;
                    let y_max;
                    if y < y_end {
                        p_top = p1;
                        p_bottom = p2;
                        y_min = y;
                        y_max = y_end;
                    } else {
                        p_top = p2;
                        p_bottom = p1;
                        y_min = y_end;
                        y_max = y;
                    };

                    if y_max < 0 || y_min >= self.height as isize {
                        continue;
                    }

                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let dxdy = dx / dy;

                    let span_start;
                    let bits_start;
                    let mut width;
                    if y_min < 0 {
                        span_start = 0;
                        bits_start = 0;
                        width = 1.0 - (p_top.x + dxdy * (0.0 - p_top.y)).fract();
                    } else {
                        let height_unsigned = 1.0 - p_top.y.fract();
                        let height = height_unsigned.copysign(dy);
                        let width1 = 1.0 - p_top.x.fract();
                        let width2 = width1 - dxdy * height_unsigned;
                        let area = 0.5 * height * (width1 + width2);

                        self.coverage[y_min as usize * self.stride + x as usize] += area;
                        self.coverage[y_min as usize * self.stride + x as usize + 1] += height - area;

                        span_start = y_min as usize + 1;
                        bits_start = y_min as usize;
                        width = width2;
                    }

                    let span_end;
                    let bits_end;
                    if y_max >= self.height as isize {
                        span_end = self.height;
                        bits_end = self.height;
                    } else {
                        span_end = y_max as usize;
                        bits_end = y_max as usize + 1;
                    }

                    for row in span_start..span_end {
                        let height = 1.0f32.copysign(dy);
                        let width1 = width;
                        let width2 = width1 - dxdy;
                        let area = 0.5 * height * (width1 + width2);

                        self.coverage[row * self.stride + x as usize] += area;
                        self.coverage[row * self.stride + x as usize + 1] += height - area;

                        width = width2;
                    }

                    if y_max < self.height as isize {
                        let height_unsigned = p_bottom.y.fract();
                        let height = height_unsigned.copysign(dy);
                        let width2 = 1.0 - p_bottom.x.fract();
                        let area = 0.5 * height * (width + width2);

                        self.coverage[y_max as usize * self.stride + x as usize] += area;
                        self.coverage[y_max as usize * self.stride + x as usize + 1] += height - area;
                    }

                    let x = x as usize;
                    for row in bits_start..bits_end {
                        let bitmask_offset = row * self.bitmasks_width;

                        let bitmask_index_1 = x >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                        let bitmask_index_2 = (x + 1) >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
                        let bit_index_1 =
                            (x >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
                        let bit_index_2 =
                            ((x + 1) >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
                        if bitmask_index_1 == bitmask_index_2 {
                            self.bitmasks[bitmask_offset + bitmask_index_1] |= (1 << bit_index_1) | (1 << bit_index_2);
                        } else {
                            self.bitmasks[bitmask_offset + bitmask_index_1] |= 1 << bit_index_1;
                            self.bitmasks[bitmask_offset + bitmask_index_2] |= 1 << bit_index_2;
                        }
                    }

                    continue;
                }

                if (x >= self.width as isize && x_end >= self.width as isize)
                    || (y >= self.height as isize && y_end >= self.height as isize)
                    || (y < 0 && y_end < 0)
                {
                    continue;
                }

                // if x == x_end && y == y_end {
                //     let height = p2.y - p1.y;
                //     let area = 0.5 * height * ((x as f32 + 1.0 - p1.x) + (x as f32 + 1.0 - p2.x));
                //     self.add_delta::<A>(x, y, height, area);
                //     continue;
                // }

                let x_inc;
                let mut x_offset;
                let x_offset_end;
                let dx;
                let area_offset;
                let area_sign;
                if p2.x > p1.x {
                    x_inc = 1;
                    x_offset = p1.x - x as f32;
                    x_offset_end = p2.x - x_end as f32;
                    dx = p2.x - p1.x;
                    area_offset = 2.0;
                    area_sign = -1.0;
                } else {
                    x_inc = -1;
                    x_offset = 1.0 - (p1.x - x as f32);
                    x_offset_end = 1.0 - (p2.x - x_end as f32);
                    dx = p1.x - p2.x;
                    area_offset = 0.0;
                    area_sign = 1.0;
                }

                let y_inc;
                let mut y_offset;
                let y_offset_end;
                let dy;
                let sign;
                if p2.y > p1.y {
                    y_inc = 1;
                    y_offset = p1.y - y as f32;
                    y_offset_end = p2.y - y_end as f32;
                    dy = p2.y - p1.y;
                    sign = 1.0;
                } else {
                    y_inc = -1;
                    y_offset = 1.0 - (p1.y - y as f32);
                    y_offset_end = 1.0 - (p2.y - y_end as f32);
                    dy = p1.y - p2.y;
                    sign = -1.0;
                }

                let dxdy = dx / dy;
                let dydx = dy / dx;

                let mut y_offset_for_prev_x = y_offset - dydx * x_offset;
                let mut x_offset_for_prev_y = x_offset - dxdy * y_offset;

                while x != x_end || y != y_end {
                    let col = x;
                    let row = y;

                    let x1 = x_offset;
                    let y1 = y_offset;

                    let x2;
                    let y2;
                    if y != y_end && (x == x_end || x_offset_for_prev_y + dxdy < 1.0) {
                        y_offset = 0.0;
                        x_offset = x_offset_for_prev_y + dxdy;
                        x_offset_for_prev_y = x_offset;
                        y_offset_for_prev_x -= 1.0;
                        y += y_inc;

                        x2 = x_offset;
                        y2 = 1.0;
                    } else {
                        x_offset = 0.0;
                        y_offset = y_offset_for_prev_x + dydx;
                        x_offset_for_prev_y -= 1.0;
                        y_offset_for_prev_x = y_offset;
                        x += x_inc;

                        x2 = 1.0;
                        y2 = y_offset;
                    }

                    let height = sign * (y2 - y1);
                    let area = 0.5 * height * (area_offset + area_sign * (x1 + x2));

                    self.add_delta::<A>(col, row, height, area);
                }

                let height = sign * (y_offset_end - y_offset);
                let area = 0.5 * height * (area_offset + area_sign * (x_offset + x_offset_end));

                self.add_delta::<A>(x, y, height, area);
            }
        })
    }

    #[inline(always)]
    fn mark_cell<A: Arch>(&mut self, x: usize, y: usize) {
        let bitmask_index = y * self.bitmasks_width + (x >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT);
        let bit_index =
            (x >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
        self.bitmasks[bitmask_index] |= 1 << bit_index;
    }

    #[inline(always)]
    fn add_delta<A: Arch>(&mut self, x: isize, y: isize, height: f32, area: f32) {
        if y < 0 || y >= self.height as isize || x >= self.width as isize {
            return;
        }

        if x < 0 {
            let coverage_index = y as usize * self.stride;
            self.coverage[coverage_index] += height;

            self.mark_cell::<A>(0, y as usize);

            return;
        }

        let coverage_index = y as usize * self.stride + x as usize;
        self.coverage[coverage_index] += area;
        self.coverage[coverage_index + 1] += height - area;

        let x = x as usize;
        let y = y as usize;
        let bitmask_offset = y * self.bitmasks_width;
        let bitmask_index_1 = x >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
        let bitmask_index_2 = (x + 1) >> Consts::<A>::PIXELS_PER_BITMASK_SHIFT;
        let bit_index_1 =
            (x >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
        let bit_index_2 =
            ((x + 1) >> Consts::<A>::PIXELS_PER_BIT_SHIFT) & (Consts::<A>::BITS_PER_BITMASK - 1);
        if bitmask_index_1 == bitmask_index_2 {
            self.bitmasks[bitmask_offset + bitmask_index_1] |= (1 << bit_index_1) | (1 << bit_index_2);
        } else {
            self.bitmasks[bitmask_offset + bitmask_index_1] |= 1 << bit_index_1;
            self.bitmasks[bitmask_offset + bitmask_index_2] |= 1 << bit_index_2;
        }
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
