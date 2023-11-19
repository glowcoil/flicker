use crate::color::Color;
use crate::flatten::{flatten, stroke};
use crate::geom::{Transform, Vec2};
use crate::path::Path;
use crate::raster::{Rasterizer, Segment};
use crate::text::Font;

const MAX_SEGMENTS: usize = 256;

pub struct Renderer {
    segments: Vec<Segment>,
    rasterizer: Rasterizer,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            segments: Vec::with_capacity(MAX_SEGMENTS),
            rasterizer: Rasterizer::new(),
        }
    }

    pub fn context<'a>(
        &'a mut self,
        data: &'a mut [u32],
        width: usize,
        height: usize,
    ) -> RenderContext<'a> {
        assert!(data.len() == width * height);

        RenderContext {
            renderer: self,
            data,
            width,
            height,
        }
    }
}

pub struct RenderContext<'a> {
    renderer: &'a mut Renderer,
    data: &'a mut [u32],
    width: usize,
    height: usize,
}

impl<'a> RenderContext<'a> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn clear(&mut self, color: Color) {
        for pixel in self.data.iter_mut() {
            *pixel = color.into();
        }
    }

    fn add_segment(&mut self, p1: Vec2, p2: Vec2) {
        self.renderer.segments.push(Segment { p1, p2 });

        if self.renderer.segments.len() == self.renderer.segments.capacity() {
            self.drain_segments();
        }
    }

    fn drain_segments(&mut self) {
        self.renderer.rasterizer.add_segments(&self.renderer.segments);
        self.renderer.segments.clear();
    }

    pub fn fill_path(&mut self, path: &Path, transform: &Transform, color: Color) {
        if path.is_empty() {
            return;
        }

        let mut min = Vec2::new(self.width as f32, self.height as f32);
        let mut max = Vec2::new(0.0, 0.0);
        for &point in &path.points {
            let transformed = transform.apply(point);
            min = min.min(transformed);
            max = max.max(transformed);
        }

        let min_x = (min.x as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y as isize).max(0).min(self.height as isize) as usize;
        let max_x = ((max.x + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Vec2::new(min_x as f32, min_y as f32);

        self.renderer.rasterizer.set_size(path_width, path_height);

        flatten(&path, transform, &mut |p1, p2| {
            self.add_segment(p1 - offset, p2 - offset);
        });

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.renderer.rasterizer.finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn stroke_path(&mut self, path: &Path, width: f32, transform: &Transform, color: Color) {
        if path.is_empty() {
            return;
        }

        let dilate_x = transform.matrix * width * Vec2::new(0.5, 0.0);
        let dilate_y = transform.matrix * width * Vec2::new(0.0, 0.5);

        let mut min = Vec2::new(self.width as f32, self.height as f32);
        let mut max = Vec2::new(0.0, 0.0);
        for &point in &path.points {
            let transformed = transform.apply(point);

            let dilate0 = transformed - dilate_x - dilate_y;
            let dilate1 = transformed + dilate_x - dilate_y;
            let dilate2 = transformed - dilate_x + dilate_y;
            let dilate3 = transformed + dilate_x + dilate_y;

            min = min.min(dilate0).min(dilate1).min(dilate2).min(dilate3);
            max = max.max(dilate0).max(dilate1).max(dilate2).max(dilate3);
        }

        let min_x = (min.x as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y as isize).max(0).min(self.height as isize) as usize;
        let max_x = ((max.x + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Vec2::new(min_x as f32, min_y as f32);

        self.renderer.rasterizer.set_size(path_width, path_height);

        stroke(&path, width, transform, &mut |p1, p2| {
            self.add_segment(p1 - offset, p2 - offset);
        });

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.renderer.rasterizer.finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn fill_text(
        &mut self,
        text: &str,
        font: &Font,
        size: f32,
        transform: &Transform,
        color: Color,
    ) {
        use rustybuzz::ttf_parser::{GlyphId, OutlineBuilder};
        use rustybuzz::UnicodeBuffer;
        use std::iter::zip;

        struct Builder {
            path: Path,
            ascent: f32,
        }

        impl OutlineBuilder for Builder {
            fn move_to(&mut self, x: f32, y: f32) {
                self.path.move_to(Vec2::new(x, self.ascent - y));
            }

            fn line_to(&mut self, x: f32, y: f32) {
                self.path.line_to(Vec2::new(x, self.ascent - y));
            }

            fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
                self.path.quadratic_to(
                    Vec2::new(x1, self.ascent - y1),
                    Vec2::new(x, self.ascent - y),
                );
            }

            fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
                self.path.cubic_to(
                    Vec2::new(x1, self.ascent - y1),
                    Vec2::new(x2, self.ascent - y2),
                    Vec2::new(x, self.ascent - y),
                );
            }

            fn close(&mut self) {
                self.path.close();
            }
        }

        let mut buf = UnicodeBuffer::new();
        buf.push_str(text);

        let scale = size / font.face.units_per_em() as f32;

        let glyphs = rustybuzz::shape(&font.face, &[], buf);

        let mut offset = 0.0;
        for (info, glyph_pos) in zip(glyphs.glyph_infos(), glyphs.glyph_positions()) {
            let mut builder = Builder {
                path: Path::new(),
                ascent: font.face.ascender() as f32,
            };
            let glyph_id = GlyphId(info.glyph_id as u16);
            font.face.outline_glyph(glyph_id, &mut builder);

            let transform = Transform::translate(
                offset + glyph_pos.x_offset as f32,
                glyph_pos.y_offset as f32,
            )
            .then(Transform::scale(scale))
            .then(*transform);
            self.fill_path(&builder.path, &transform, color);

            offset += glyph_pos.x_advance as f32;
        }
    }
}
