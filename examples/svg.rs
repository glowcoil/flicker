use std::time::Duration;

use flicker::{Color, Font, Renderer, Transform, Vec2};
use portlight::{
    App, Bitmap, Event, MouseButton, Point, Response, Size, WindowContext, WindowOptions,
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
    renderer: Renderer,
    framebuffer: Vec<u32>,
    font: Font,
    commands: Vec<svg::Command>,
    timer: FrameTimer,
    mouse_pos: Point,
    dragging: bool,
    transform: Transform,
}

impl State {
    fn new(commands: Vec<svg::Command>) -> State {
        State {
            renderer: Renderer::new(),
            framebuffer: Vec::new(),
            font: Font::from_bytes(include_bytes!("res/SourceSansPro-Regular.otf"), 0).unwrap(),
            commands,
            timer: FrameTimer::new(),
            mouse_pos: Point { x: -1.0, y: -1.0 },
            dragging: false,
            transform: Transform::id(),
        }
    }

    fn handle_event(&mut self, cx: &WindowContext, event: Event) -> Response {
        match event {
            Event::Frame => {
                let scale = cx.window().scale();
                let width = (WIDTH as f64 * scale) as usize;
                let height = (HEIGHT as f64 * scale) as usize;

                self.framebuffer.resize(width as usize * height as usize, 0xFF000000);

                let mut context = self.renderer.context(&mut self.framebuffer, width, height);

                context.clear(Color::rgba(255, 255, 255, 255));

                let time = std::time::Instant::now();
                svg::render(
                    &self.commands,
                    &self.transform.then(Transform::scale(scale as f32)),
                    &mut context,
                );
                let elapsed = time.elapsed();

                self.timer.update(elapsed);

                context.fill_text(
                    &format!("{:#.3?}", self.timer.average()),
                    &self.font,
                    24.0,
                    &Transform::scale(scale as f32),
                    Color::rgba(0, 0, 0, 255),
                );

                cx.window().present(Bitmap::new(&self.framebuffer, width, height));
            }
            Event::MouseMove(pos) => {
                if self.dragging {
                    let prev = Vec2::new(self.mouse_pos.x as f32, self.mouse_pos.y as f32);
                    let curr = Vec2::new(pos.x as f32, pos.y as f32);
                    self.transform =
                        self.transform.then(Transform::translate(curr.x - prev.x, curr.y - prev.y));
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
                let width = WIDTH as f32;
                let height = HEIGHT as f32;

                self.transform = self
                    .transform
                    .then(Transform::translate(-0.5 * width, -0.5 * height))
                    .then(Transform::scale(1.02f32.powf(delta.y as f32)))
                    .then(Transform::translate(0.5 * width, 0.5 * height));

                return Response::Capture;
            }
            Event::Close => {
                cx.app().exit();
            }
            _ => {}
        }

        Response::Ignore
    }
}

fn main() {
    let path_arg = std::env::args().nth(1);
    let path = path_arg.as_ref().map(|s| &s[..]).unwrap_or("examples/res/tiger.svg");
    let commands = svg::from_file(path).unwrap();

    let app = App::new().unwrap();

    let mut state = State::new(commands);

    let window = WindowOptions::new()
        .title("flicker")
        .size(Size::new(WIDTH as f64, HEIGHT as f64))
        .open(app.handle(), move |cx, event| state.handle_event(cx, event))
        .unwrap();

    window.show();

    app.run().unwrap();
}
