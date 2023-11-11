use std::slice;

use crate::geom::*;

const TOLERANCE: f32 = 0.1;

#[derive(Clone)]
pub struct Path {
    verbs: Vec<Verb>,
    points: Vec<Vec2>,
}

#[derive(Copy, Clone)]
pub enum Verb {
    Move,
    Line,
    Quadratic,
    Cubic,
    Close,
}

#[derive(Copy, Clone)]
pub enum Command {
    Move(Vec2),
    Line(Vec2),
    Quadratic(Vec2, Vec2),
    Cubic(Vec2, Vec2, Vec2),
    Close,
}

impl Path {
    #[inline]
    pub fn new() -> Path {
        Path {
            verbs: Vec::new(),
            points: Vec::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    #[inline]
    pub fn bounds(&self, transform: &Transform) -> (Vec2, Vec2) {
        if self.is_empty() {
            return (Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0));
        }

        let first = transform.apply(*self.points.first().unwrap());
        let mut min = first;
        let mut max = first;
        for &point in self.points[1..].iter() {
            let transformed = transform.apply(point);
            min = min.min(transformed);
            max = max.max(transformed);
        }

        (min, max)
    }

    #[inline]
    pub fn move_to(&mut self, point: Vec2) -> &mut Self {
        self.verbs.push(Verb::Move);
        self.points.push(point);
        self
    }

    #[inline]
    pub fn line_to(&mut self, point: Vec2) -> &mut Self {
        self.verbs.push(Verb::Line);
        self.points.push(point);
        self
    }

    #[inline]
    pub fn quadratic_to(&mut self, control: Vec2, point: Vec2) -> &mut Self {
        self.verbs.push(Verb::Quadratic);
        self.points.push(control);
        self.points.push(point);
        self
    }

    #[inline]
    pub fn cubic_to(&mut self, control1: Vec2, control2: Vec2, point: Vec2) -> &mut Self {
        self.verbs.push(Verb::Cubic);
        self.points.push(control1);
        self.points.push(control2);
        self.points.push(point);
        self
    }

    #[inline]
    pub fn arc(&mut self, radius: f32, start_angle: f32, end_angle: f32) -> &mut Self {
        let mut last = self.points.last().cloned().unwrap_or(Vec2::new(0.0, 0.0));
        let mut vector = Vec2::new(start_angle.cos(), start_angle.sin());
        let mut angle = 0.0;

        let center = last - radius * vector;
        let winding = if end_angle < start_angle { -1.0 } else { 1.0 };
        let total_angle = (end_angle - start_angle).abs();

        // approximate quarter-circle arcs with cubics
        let quarter_circle = 0.5 * std::f32::consts::PI;
        let k = (4.0 / 3.0) * (0.25 * quarter_circle).tan();
        while angle + quarter_circle < total_angle {
            let tangent = winding * Vec2::new(-vector.y, vector.x);

            let control1 = last + radius * k * tangent;
            let point = center + radius * tangent;
            let control2 = point + radius * k * vector;
            self.cubic_to(control1, control2, point);

            angle += quarter_circle;
            vector = tangent;
            last = point;
        }

        // approximate the remainder of the arc with a single cubic
        let tangent = winding * Vec2::new(-vector.y, vector.x);
        let angle_size = total_angle - angle;
        let k = (4.0 / 3.0) * (0.25 * angle_size).tan();

        let end_vector = Vec2::new(end_angle.cos(), end_angle.sin());
        let end_tangent = winding * Vec2::new(-end_vector.y, end_vector.x);

        let control1 = last + radius * k * tangent;
        let point = center + radius * end_vector;
        let control2 = point - radius * k * end_tangent;
        self.cubic_to(control1, control2, point);

        self
    }

    #[inline]
    pub fn close(&mut self) -> &mut Self {
        self.verbs.push(Verb::Close);
        self
    }

    #[inline]
    pub fn push(&mut self, command: Command) {
        match command {
            Command::Move(point) => {
                self.move_to(point);
            }
            Command::Line(point) => {
                self.line_to(point);
            }
            Command::Quadratic(control, point) => {
                self.quadratic_to(control, point);
            }
            Command::Cubic(control1, control2, point) => {
                self.cubic_to(control1, control2, point);
            }
            Command::Close => {
                self.close();
            }
        }
    }

    #[inline]
    pub fn segments(&self) -> Segments {
        Segments {
            first: Vec2::new(0.0, 0.0),
            prev: Vec2::new(0.0, 0.0),
            verbs: self.verbs.iter(),
            points: self.points.iter(),
        }
    }

    #[inline]
    pub(crate) fn stroke(&self, width: f32, transform: &Transform) -> Path {
        #[inline]
        fn join(path: &mut Path, width: f32, prev_normal: Vec2, next_normal: Vec2, point: Vec2) {
            let offset = 1.0 / (1.0 + prev_normal.dot(next_normal));
            if offset.abs() > 2.0 {
                path.line_to(point + 0.5 * width * prev_normal);
                path.line_to(point + 0.5 * width * next_normal);
            } else {
                path.line_to(point + 0.5 * width * offset * (prev_normal + next_normal));
            }
        }

        #[inline]
        fn offset(path: &mut Path, width: f32, contour: &[Vec2], closed: bool, reverse: bool) {
            let first_point = if closed == reverse {
                contour[0]
            } else {
                *contour.last().unwrap()
            };
            let mut prev_point = first_point;
            let mut prev_normal = Vec2::new(0.0, 0.0);
            let mut i = 0;
            loop {
                let next_point = if i < contour.len() {
                    if reverse {
                        contour[contour.len() - i - 1]
                    } else {
                        contour[i]
                    }
                } else {
                    first_point
                };

                if next_point != prev_point || i == contour.len() {
                    let next_tangent = next_point - prev_point;
                    let next_normal = Vec2::new(-next_tangent.y, next_tangent.x);
                    let next_normal_len = next_normal.length();
                    let next_normal = if next_normal_len == 0.0 {
                        Vec2::new(0.0, 0.0)
                    } else {
                        next_normal * (1.0 / next_normal_len)
                    };

                    join(path, width, prev_normal, next_normal, prev_point);

                    prev_point = next_point;
                    prev_normal = next_normal;
                }

                i += 1;
                if i > contour.len() {
                    break;
                }
            }
        }

        let mut flattened = Path::new();
        let mut prev = Vec2::new(0.0, 0.0);
        for segment in self.segments() {
            segment.flatten(transform, |p1, p2| {
                if prev != p1 {
                    flattened.push(Command::Move(p1));
                }
                flattened.push(Command::Line(p2));
                prev = p2
            });
        }

        // This is only valid for isotropic transforms.
        // FIXME: Handle arbitrary transforms properly.
        let width = transform.matrix.determinant().abs().sqrt() * width;

        let mut path = Path::new();

        let mut contour_start = 0;
        let mut contour_end = 0;
        let mut closed = false;
        let mut verbs = flattened.verbs.iter();
        loop {
            let verb = verbs.next();

            if let Some(Verb::Close) = verb {
                closed = true;
            }

            if let None | Some(Verb::Move) | Some(Verb::Close) = verb {
                if contour_start != contour_end {
                    let contour = &flattened.points[contour_start..contour_end];

                    let base = path.verbs.len();
                    offset(&mut path, width, contour, closed, false);
                    path.verbs[base] = Verb::Move;
                    if closed {
                        path.close();
                    }

                    let base = path.verbs.len();
                    offset(&mut path, width, contour, closed, true);
                    if closed {
                        path.verbs[base] = Verb::Move;
                    }
                    path.close();
                }
            }

            if let Some(verb) = verb {
                match verb {
                    Verb::Move => {
                        contour_start = contour_end;
                        contour_end = contour_start + 1;
                    }
                    Verb::Line => {
                        contour_end += 1;
                    }
                    Verb::Close => {
                        contour_start = contour_end;
                        contour_end = contour_start;
                        closed = true;
                    }
                    _ => {
                        unreachable!();
                    }
                }
            } else {
                break;
            }
        }

        path
    }
}

pub enum Segment {
    Line(Vec2, Vec2),
    Quadratic(Vec2, Vec2, Vec2),
    Cubic(Vec2, Vec2, Vec2, Vec2),
}

impl Segment {
    #[inline]
    pub(crate) fn flatten(&self, transform: &Transform, mut sink: impl FnMut(Vec2, Vec2)) {
        match *self {
            Segment::Line(p1, p2) => {
                sink(transform.apply(p1), transform.apply(p2));
            }
            Segment::Quadratic(p1, p2, p3) => {
                let p1 = transform.apply(p1);
                let p2 = transform.apply(p2);
                let p3 = transform.apply(p3);

                let dt = ((4.0 * TOLERANCE) / (p1 - 2.0 * p2 + p3).length()).sqrt();

                let mut prev = p1;
                let mut t = 0.0;
                while t < 1.0 {
                    t = (t + dt).min(1.0);

                    let p01 = Vec2::lerp(t, p1, p2);
                    let p12 = Vec2::lerp(t, p2, p3);
                    let point = Vec2::lerp(t, p01, p12);

                    sink(prev, point);
                    prev = point;
                }
            }
            Segment::Cubic(p1, p2, p3, p4) => {
                let p1 = transform.apply(p1);
                let p2 = transform.apply(p2);
                let p3 = transform.apply(p3);
                let p4 = transform.apply(p4);

                let a = -1.0 * p1 + 3.0 * p2 - 3.0 * p3 + p4;
                let b = 3.0 * (p1 - 2.0 * p2 + p3);
                let conc = b.length().max((a + b).length());
                let dt = ((8.0f32.sqrt() * TOLERANCE) / conc).sqrt();

                let mut prev = p1;
                let mut t = 0.0;
                while t < 1.0 {
                    t = (t + dt).min(1.0);

                    let p01 = Vec2::lerp(t, p1, p2);
                    let p12 = Vec2::lerp(t, p2, p3);
                    let p23 = Vec2::lerp(t, p3, p4);
                    let p012 = Vec2::lerp(t, p01, p12);
                    let p123 = Vec2::lerp(t, p12, p23);

                    let point = Vec2::lerp(t, p012, p123);

                    sink(prev, point);
                    prev = point;
                }
            }
        }
    }
}

pub struct Segments<'a> {
    first: Vec2,
    prev: Vec2,
    verbs: slice::Iter<'a, Verb>,
    points: slice::Iter<'a, Vec2>,
}

impl<'a> Iterator for Segments<'a> {
    type Item = Segment;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(verb) = self.verbs.next() {
            match *verb {
                Verb::Move => {
                    self.first = *self.points.next().unwrap();
                    self.prev = self.first;
                }
                Verb::Line => {
                    let prev = self.prev;
                    let point = *self.points.next().unwrap();
                    self.prev = point;

                    return Some(Segment::Line(prev, point));
                }
                Verb::Quadratic => {
                    let prev = self.prev;
                    let control = *self.points.next().unwrap();
                    let point = *self.points.next().unwrap();
                    self.prev = point;

                    return Some(Segment::Quadratic(prev, control, point));
                }
                Verb::Cubic => {
                    let prev = self.prev;
                    let control1 = *self.points.next().unwrap();
                    let control2 = *self.points.next().unwrap();
                    let point = *self.points.next().unwrap();
                    self.prev = point;

                    return Some(Segment::Cubic(prev, control1, control2, point));
                }
                Verb::Close => {
                    let prev = self.prev;
                    self.prev = self.first;
                    if prev != self.first {
                        return Some(Segment::Line(prev, self.first));
                    }
                }
            }
        }

        None
    }
}
