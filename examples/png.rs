use flicker::{Canvas, Color, Mat2x2, Path, Transform, Vec2};

enum Style {
    Fill,
    Stroke(f32),
}

struct Command {
    path: Path,
    style: Style,
    color: Color,
}

fn main() {
    let mut canvas = Canvas::with_size(1920, 1080);

    let path = std::env::args().nth(1).expect("provide an svg file");
    let tree = usvg::Tree::from_file(path, &usvg::Options::default()).unwrap();

    fn build_list(node: &usvg::Node, commands: &mut Vec<Command>) {
        use usvg::NodeExt;
        match *node.borrow() {
            usvg::NodeKind::Path(ref p) => {
                let t = node.transform();
                let transform = Transform::new(
                    Mat2x2::new(t.a as f32, t.c as f32, t.b as f32, t.d as f32),
                    Vec2::new(t.e as f32, t.f as f32),
                )
                .then(Transform::scale(1.0));

                let mut path = Path::new();
                for segment in p.data.0.iter() {
                    match *segment {
                        usvg::PathSegment::MoveTo { x, y } => {
                            path.move_to(
                                Vec2::new(500.0, 0.0)
                                    + transform.apply(1.0 * Vec2::new(x as f32, y as f32)),
                            );
                        }
                        usvg::PathSegment::LineTo { x, y } => {
                            path.line_to(
                                Vec2::new(500.0, 0.0)
                                    + transform.apply(1.0 * Vec2::new(x as f32, y as f32)),
                            );
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
                                Vec2::new(500.0, 0.0)
                                    + transform.apply(1.0 * Vec2::new(x1 as f32, y1 as f32)),
                                Vec2::new(500.0, 0.0)
                                    + transform.apply(1.0 * Vec2::new(x2 as f32, y2 as f32)),
                                Vec2::new(500.0, 0.0)
                                    + transform.apply(1.0 * Vec2::new(x as f32, y as f32)),
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
                        commands.push(Command { path: path.clone(), style: Style::Fill, color });
                    }
                }

                if let Some(ref stroke) = p.stroke {
                    if let usvg::Paint::Color(color) = stroke.paint {
                        let color =
                            Color::rgba(color.red, color.green, color.blue, stroke.opacity.to_u8());
                        commands.push(Command { path, style: Style::Stroke(stroke.width.value() as f32), color });
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

    let mut commands = Vec::new();
    build_list(&tree.root(), &mut commands);
    let time = std::time::Instant::now();
    render(&commands, &mut canvas);
    dbg!(time.elapsed());

    use png::HasParameters;
    use std::fs::File;
    use std::io::BufWriter;

    let path = std::path::Path::new(r"out.png");
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, 1920, 1080);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();

    let mut data = vec![0; 4 * 1920 * 1080];
    let mut i = 0;
    for pixel in canvas.data() {
        data[4 * i] = ((pixel >> 16) & 0xFF) as u8;
        data[4 * i + 1] = ((pixel >> 8) & 0xFF) as u8;
        data[4 * i + 2] = ((pixel >> 0) & 0xFF) as u8;
        data[4 * i + 3] = ((pixel >> 24) & 0xFF) as u8;
        i += 1;
    }

    writer.write_image_data(&data[..]).unwrap();
}
