use std::cell::{Cell, RefCell};
use std::time::Duration;

use flicker::{Canvas, Color, Font, Transform, Vec2};
use portlight::{Application, MouseButton, Point, Rect, Window, WindowHandler, WindowOptions};

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

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

struct Handler {
    canvas: RefCell<Canvas>,
    font: Font,
    commands: Vec<svg::Command>,
    timer: RefCell<FrameTimer>,
    mouse_pos: Cell<Point>,
    dragging: Cell<bool>,
    transform: Cell<Transform>,
}

impl Handler {
    fn new(commands: Vec<svg::Command>) -> Handler {
        Handler {
            canvas: RefCell::new(Canvas::with_size(WIDTH, HEIGHT)),
            font: Font::from_bytes(include_bytes!("res/SourceSansPro-Regular.otf"), 0).unwrap(),
            commands,
            timer: RefCell::new(FrameTimer::new()),
            mouse_pos: Cell::new(Point { x: -1.0, y: -1.0 }),
            dragging: Cell::new(false),
            transform: Cell::new(Transform::id()),
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
        svg::render(
            &self.commands,
            &self.transform.get(),
            &mut *self.canvas.borrow_mut(),
        );
        let elapsed = time.elapsed();

        self.timer.borrow_mut().update(elapsed);

        self.canvas.borrow_mut().fill_text(
            &format!("{:#.3?}", self.timer.borrow().average()),
            &self.font,
            24.0,
            &Transform::id(),
            Color::rgba(0, 0, 0, 255),
        );

        window.update_contents(
            self.canvas.borrow().data(),
            self.canvas.borrow().width(),
            self.canvas.borrow().height(),
        );
    }

    fn mouse_move(&self, _window: &Window, position: Point) {
        if self.dragging.get() {
            let prev = Vec2::new(self.mouse_pos.get().x as f32, self.mouse_pos.get().y as f32);
            let curr = Vec2::new(position.x as f32, position.y as f32);
            self.transform.set(
                self.transform
                    .get()
                    .then(Transform::translate(curr.x - prev.x, curr.y - prev.y)),
            );
        }

        self.mouse_pos.set(position);
    }

    fn mouse_down(&self, _window: &Window, button: MouseButton) -> bool {
        if button == MouseButton::Left {
            self.dragging.set(true);
        }

        true
    }

    fn mouse_up(&self, _window: &Window, button: MouseButton) -> bool {
        if button == MouseButton::Left {
            self.dragging.set(false);
        }

        true
    }

    fn scroll(&self, _window: &Window, _dx: f64, dy: f64) -> bool {
        let mouse_pos = self.mouse_pos.get();

        self.transform.set(
            self.transform
                .get()
                .then(Transform::translate(
                    -mouse_pos.x as f32,
                    -mouse_pos.y as f32,
                ))
                .then(Transform::scale(1.02f32.powf(dy as f32)))
                .then(Transform::translate(mouse_pos.x as f32, mouse_pos.y as f32)),
        );

        true
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
    let commands = svg::from_file(path).unwrap();

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
