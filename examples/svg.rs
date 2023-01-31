use std::cell::RefCell;
use std::time::Duration;

use flicker::{Canvas, Color, Font, Mat2x2, Path, Transform, Vec2};
use portlight::{Application, Rect, Window, WindowHandler, WindowOptions};

const WIDTH: usize = 768;
const HEIGHT: usize = 768;

enum Style {
    Fill,
    Stroke(f32),
}

struct Command {
    path: Path,
    style: Style,
    color: Color,
}

fn build_list(node: &usvg::Node, commands: &mut Vec<Command>) {
    use usvg::NodeExt;
    match *node.borrow() {
        usvg::NodeKind::Path(ref p) => {
            let t = node.transform();
            let transform = Transform::new(
                Mat2x2::new(t.a as f32, t.c as f32, t.b as f32, t.d as f32),
                Vec2::new(t.e as f32, t.f as f32),
            );

            let mut path = Path::new();
            for segment in p.data.0.iter() {
                match *segment {
                    usvg::PathSegment::MoveTo { x, y } => {
                        path.move_to(transform.apply(1.0 * Vec2::new(x as f32, y as f32)));
                    }
                    usvg::PathSegment::LineTo { x, y } => {
                        path.line_to(transform.apply(1.0 * Vec2::new(x as f32, y as f32)));
                    }
                    usvg::PathSegment::CurveTo {
                        x1,
                        y1,
                        x2,
                        y2,
                        x,
                        y,
                    } => {
                        path.cubic_to(
                            transform.apply(1.0 * Vec2::new(x1 as f32, y1 as f32)),
                            transform.apply(1.0 * Vec2::new(x2 as f32, y2 as f32)),
                            transform.apply(1.0 * Vec2::new(x as f32, y as f32)),
                        );
                    }
                    usvg::PathSegment::ClosePath => {
                        path.close();
                    }
                }
            }

            if let Some(ref fill) = p.fill {
                if let usvg::Paint::Color(color) = fill.paint {
                    let color =
                        Color::rgba(color.red, color.green, color.blue, fill.opacity.to_u8());
                    commands.push(Command {
                        path: path.clone(),
                        style: Style::Fill,
                        color,
                    });
                }
            }

            if let Some(ref stroke) = p.stroke {
                if let usvg::Paint::Color(color) = stroke.paint {
                    let color =
                        Color::rgba(color.red, color.green, color.blue, stroke.opacity.to_u8());
                    commands.push(Command {
                        path,
                        style: Style::Stroke(stroke.width.value() as f32),
                        color,
                    });
                }
            }
        }
        _ => {}
    }

    for child in node.children() {
        build_list(&child, commands);
    }
}

fn render(commands: &[Command], canvas: &mut Canvas) {
    for command in commands {
        match command.style {
            Style::Fill => {
                canvas.fill_path(&command.path, command.color);
            }
            Style::Stroke(width) => {
                canvas.stroke_path(&command.path, width, command.color);
            }
        }
    }
}

const AVERAGE_WINDOW: usize = 64;

struct FrameTimer {
    times: Vec<Duration>,
    valid: usize,
    time_index: usize,
    running_sum: Duration,
}

impl FrameTimer {
    fn new() -> FrameTimer {
        FrameTimer {
            times: vec![Duration::default(); AVERAGE_WINDOW],
            valid: 0,
            time_index: 0,
            running_sum: Duration::default(),
        }
    }

    fn update(&mut self, time: Duration) {
        self.times[self.time_index] = time;

        self.time_index = (self.time_index + 1) % AVERAGE_WINDOW;
        if self.valid < AVERAGE_WINDOW {
            self.valid += 1;
        }

        self.running_sum += time;
        if self.valid == AVERAGE_WINDOW {
            self.running_sum -= self.times[self.time_index];
        }
    }

    fn average(&self) -> Duration {
        self.running_sum.div_f64(self.valid as f64)
    }
}

struct Handler {
    canvas: RefCell<Canvas>,
    font: Font,
    commands: Vec<Command>,
    timer: RefCell<FrameTimer>,
}

impl Handler {
    fn new(commands: Vec<Command>) -> Handler {
        Handler {
            canvas: RefCell::new(Canvas::with_size(WIDTH, HEIGHT)),
            font: Font::from_bytes(include_bytes!("res/SourceSansPro-Regular.otf"), 0).unwrap(),
            commands,
            timer: RefCell::new(FrameTimer::new()),
        }
    }
}

impl WindowHandler for Handler {
    fn frame(&self, window: &Window) {
        window.request_display();
    }

    fn display(&self, window: &Window) {
        self.canvas
            .borrow_mut()
            .clear(Color::rgba(255, 255, 255, 255));

        let time = std::time::Instant::now();
        render(&self.commands, &mut *self.canvas.borrow_mut());
        let elapsed = time.elapsed();

        self.timer.borrow_mut().update(elapsed);

        self.canvas.borrow_mut().fill_text(
            &format!("{:#.3?}", self.timer.borrow().average()),
            &self.font,
            24.0,
            Color::rgba(0, 0, 0, 255),
        );

        window.update_contents(
            self.canvas.borrow().data(),
            self.canvas.borrow().width(),
            self.canvas.borrow().height(),
        );
    }

    fn request_close(&self, window: &Window) {
        window.close().unwrap();
        window.application().stop();
    }
}

fn main() {
    let path_arg = std::env::args().nth(1);
    let path = path_arg
        .as_ref()
        .map(|s| &s[..])
        .unwrap_or("examples/res/tiger.svg");
    let tree = usvg::Tree::from_file(path, &usvg::Options::default()).unwrap();

    let mut commands = Vec::new();
    build_list(&tree.root(), &mut commands);

    let app = Application::new().unwrap();

    Window::open(
        &app,
        WindowOptions {
            title: "flicker".to_string(),
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: WIDTH as f64,
                height: HEIGHT as f64,
            },
            handler: Box::new(Handler::new(commands)),
            ..WindowOptions::default()
        },
    )
    .unwrap();

    app.start().unwrap();
}
