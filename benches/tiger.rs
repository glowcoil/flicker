use criterion::{criterion_group, criterion_main, Criterion};

use flicker::Canvas;

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut canvas = Canvas::with_size(WIDTH, HEIGHT);

    let commands = svg::from_file("examples/res/tiger.svg").unwrap();

    c.bench_function("tiger", |b| {
        b.iter(|| {
            svg::render(&commands, &mut canvas);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
