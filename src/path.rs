use crate::geom::*;

#[derive(Clone)]
pub struct Path {
    pub(crate) verbs: Vec<Verb>,
    pub(crate) points: Vec<Vec2>,
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
}
