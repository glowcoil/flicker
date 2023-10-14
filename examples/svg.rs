use std::time::Duration;

use flicker::{Canvas, Color, Font, Transform, Vec2};
use portlight::{
    AppContext, AppOptions, Bitmap, Event, MouseButton, Point, Response, Size, Window,
    WindowOptions,
};

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

const AVERAGE_WINDOW: usize = 32;

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
        self.running_sum += time;
        if self.valid == AVERAGE_WINDOW {
            self.running_sum -= self.times[self.time_index];
        }

        self.times[self.time_index] = time;
        self.time_index = (self.time_index + 1) % AVERAGE_WINDOW;

        if self.valid < AVERAGE_WINDOW {
            self.valid += 1;
        }
    }

    fn average(&self) -> Duration {
        self.running_sum.div_f64(self.valid as f64)
    }
}

struct State {
    scale: f64,
    canvas: Canvas,
    font: Font,
    commands: Vec<svg::Command>,
    timer: FrameTimer,
    mouse_pos: Point,
    dragging: bool,
    transform: Transform,
    window: Window,
}

impl State {
    fn new(commands: Vec<svg::Command>, window: Window) -> State {
        let scale = window.scale();
        let width = WIDTH as f64 * scale;
        let height = HEIGHT as f64 * scale;

        State {
            scale,
            canvas: Canvas::with_size(width as usize, height as usize),
            font: Font::from_bytes(include_bytes!("res/SourceSansPro-Regular.otf"), 0).unwrap(),
            commands,
            timer: FrameTimer::new(),
            mouse_pos: Point { x: -1.0, y: -1.0 },
            dragging: false,
            transform: Transform::id(),
            window,
        }
    }

    fn handle_event(&mut self, cx: &AppContext<Self>, event: Event) -> Response {
        match event {
            Event::Frame => {
                self.canvas.clear(Color::rgba(255, 255, 255, 255));

                let time = std::time::Instant::now();
                svg::render(
                    &self.commands,
                    &self.transform.then(Transform::scale(self.scale as f32)),
                    &mut self.canvas,
                );
                let elapsed = time.elapsed();

                self.timer.update(elapsed);

                self.canvas.fill_text(
                    &format!("{:#.3?}", self.timer.average()),
                    &self.font,
                    24.0,
                    &Transform::scale(self.scale as f32),
                    Color::rgba(0, 0, 0, 255),
                );

                self.window.present(Bitmap::new(
                    self.canvas.data(),
                    self.canvas.width(),
                    self.canvas.height(),
                ));
            }
            Event::MouseMove(pos) => {
                if self.dragging {
                    let prev = Vec2::new(self.mouse_pos.x as f32, self.mouse_pos.y as f32);
                    let curr = Vec2::new(pos.x as f32, pos.y as f32);
                    self.transform = self
                        .transform
                        .then(Transform::translate(curr.x - prev.x, curr.y - prev.y));
                }

                self.mouse_pos = pos;
            }
            Event::MouseDown(btn) => {
                if btn == MouseButton::Left {
                    self.dragging = true;
                }

                return Response::Capture;
            }
            Event::MouseUp(btn) => {
                if btn == MouseButton::Left {
                    self.dragging = false;
                }

                return Response::Capture;
            }
            Event::Scroll(delta) => {
                self.transform = self
                    .transform
                    .then(Transform::translate(
                        -self.mouse_pos.x as f32,
                        -self.mouse_pos.y as f32,
                    ))
                    .then(Transform::scale(1.02f32.powf(delta.y as f32)))
                    .then(Transform::translate(
                        self.mouse_pos.x as f32,
                        self.mouse_pos.y as f32,
                    ));

                return Response::Capture;
            }
            Event::Close => {
                cx.exit();
            }
            _ => {}
        }

        Response::Ignore
    }
}

fn main() {
    let path_arg = std::env::args().nth(1);
    let path = path_arg
        .as_ref()
        .map(|s| &s[..])
        .unwrap_or("examples/res/tiger.svg");
    let commands = svg::from_file(path).unwrap();

    let mut app = AppOptions::new()
        .build(|cx| {
            let window = WindowOptions::new()
                .title("flicker")
                .size(Size::new(WIDTH as f64, HEIGHT as f64))
                .open(cx, State::handle_event)
                .unwrap();

            let scale = window.scale();
            let width = (WIDTH as f64 * scale) as usize;
            let height = (HEIGHT as f64 * scale) as usize;
            let framebuffer = vec![0xFFFF00FF; width * height];
            window.present(Bitmap::new(&framebuffer, width, height));

            window.show();

            Ok(State::new(commands, window))
        })
        .unwrap();

    app.run().unwrap();
}
