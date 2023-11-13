use crate::color::Color;
use crate::geom::{Transform, Vec2};
use crate::path::Path;
use crate::raster::{Rasterizer, Segment};
use crate::text::Font;

const MAX_SEGMENTS: usize = 256;

pub struct Canvas {
    width: usize,
    height: usize,
    data: Vec<u32>,
    segments: Vec<Segment>,
    rasterizer: Rasterizer,
}

impl Canvas {
    pub fn with_size(width: usize, height: usize) -> Canvas {
        Canvas {
            width,
            height,
            data: vec![0xFF000000; width * height],
            segments: Vec::with_capacity(MAX_SEGMENTS),
            rasterizer: Rasterizer::with_size(width, height),
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[u32] {
        &self.data[0..self.width * self.height]
    }

    pub fn clear(&mut self, color: Color) {
        for pixel in self.data.iter_mut() {
            *pixel = color.into();
        }
    }

    fn add_segment(&mut self, p1: Vec2, p2: Vec2) {
        self.segments.push(Segment { p1, p2 });

        if self.segments.len() == self.segments.capacity() {
            self.drain_segments();
        }
    }

    fn drain_segments(&mut self) {
        self.rasterizer.add_segments(&self.segments);
        self.segments.clear();
    }

    pub fn fill_path(&mut self, path: &Path, transform: &Transform, color: Color) {
        if path.is_empty() {
            return;
        }

        let (min, max) = path.bounds(transform);

        let min_x = (min.x as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y as isize).max(0).min(self.width as isize) as usize;
        let max_x = ((max.x + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Vec2::new(min_x as f32, min_y as f32);

        self.rasterizer.set_size(path_width, path_height);

        for segment in path.segments() {
            segment.flatten(transform, |p1, p2| {
                self.add_segment(p1 - offset, p2 - offset);
            });
        }

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.rasterizer
            .finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn stroke_path(&mut self, path: &Path, width: f32, transform: &Transform, color: Color) {
        if path.is_empty() {
            return;
        }

        let (mut min, mut max) = path.bounds(transform);
        min.x -= 0.5 * width;
        min.y -= 0.5 * width;
        max.x += 0.5 * width;
        max.y += 0.5 * width;

        let min_x = (min.x.floor() as isize).max(0).min(self.width as isize) as usize;
        let min_y = (min.y.floor() as isize).max(0).min(self.width as isize) as usize;
        let max_x = ((max.x.ceil() + 1.0) as isize).max(0).min(self.width as isize) as usize;
        let max_y = ((max.y.ceil() + 1.0) as isize).max(0).min(self.height as isize) as usize;

        if max_x <= min_x || max_y <= min_y {
            return;
        }

        let path_width = max_x - min_x;
        let path_height = max_y - min_y;

        let offset = Vec2::new(min_x as f32, min_y as f32);

        self.rasterizer.set_size(path_width, path_height);

        path.stroke(width, transform, |p1, p2| {
            self.add_segment(p1 - offset, p2 - offset);
        });

        self.drain_segments();

        let data_start = min_y * self.width + min_x;
        self.rasterizer
            .finish(color, &mut self.data[data_start..], self.width);
    }

    pub fn fill_text(
        &mut self,
        text: &str,
        font: &Font,
        size: f32,
        transform: &Transform,
        color: Color,
    ) {
        use swash::scale::*;
        use swash::shape::*;
        use zeno::*;

        let mut shape_context = ShapeContext::new();
        let mut shaper = shape_context.builder(font.as_ref()).size(size).build();

        let mut scale_context = ScaleContext::new();
        let mut scaler = scale_context.builder(font.as_ref()).size(size).build();

        let mut offset = 1.0;
        shaper.add_str(text);
        shaper.shape_with(|cluster| {
            for glyph in cluster.glyphs {
                if let Some(outline) = scaler.scale_outline(glyph.id) {
                    let mut path = Path::new();

                    let mut points = outline.points().iter();
                    for verb in outline.verbs() {
                        match verb {
                            Verb::MoveTo => {
                                let point = points.next().unwrap();
                                path.move_to(Vec2::new(point.x + offset, -point.y + size));
                            }
                            Verb::LineTo => {
                                let point = points.next().unwrap();
                                path.line_to(Vec2::new(point.x + offset, -point.y + size));
                            }
                            Verb::CurveTo => {
                                let control1 = points.next().unwrap();
                                let control2 = points.next().unwrap();
                                let point = points.next().unwrap();
                                path.cubic_to(
                                    Vec2::new(control1.x + offset, -control1.y + size),
                                    Vec2::new(control2.x + offset, -control2.y + size),
                                    Vec2::new(point.x + offset, -point.y + size),
                                );
                            }
                            Verb::QuadTo => {
                                let control = points.next().unwrap();
                                let point = points.next().unwrap();
                                path.quadratic_to(
                                    Vec2::new(control.x + offset, -control.y + size),
                                    Vec2::new(point.x + offset, -point.y + size),
                                );
                            }
                            Verb::Close => {
                                path.close();
                            }
                        }
                    }

                    self.fill_path(&path, transform, color);

                    offset += glyph.advance;
                }
            }
        });
    }
}
