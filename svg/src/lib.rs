use std::error::Error;
use std::path;

use flicker::{Canvas, Color, Mat2x2, Path, Transform, Vec2};

pub enum Style {
    Fill,
    Stroke(f32),
}

pub struct Command {
    pub path: Path,
    pub style: Style,
    pub color: Color,
}

pub fn from_file<P: AsRef<path::Path>>(path: P) -> Result<Vec<Command>, Box<dyn Error>> {
    let tree = usvg::Tree::from_file(path, &usvg::Options::default())?;

    let mut commands = Vec::new();
    build_list(&tree.root(), &mut commands);

    Ok(commands)
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
                        path.move_to(transform.apply(Vec2::new(x as f32, y as f32)));
                    }
                    usvg::PathSegment::LineTo { x, y } => {
                        path.line_to(transform.apply(Vec2::new(x as f32, y as f32)));
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
                            transform.apply(Vec2::new(x1 as f32, y1 as f32)),
                            transform.apply(Vec2::new(x2 as f32, y2 as f32)),
                            transform.apply(Vec2::new(x as f32, y as f32)),
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

pub fn render(commands: &[Command], transform: &Transform, canvas: &mut Canvas) {
    for command in commands {
        match command.style {
            Style::Fill => {
                canvas.fill_path(&command.path, transform, command.color);
            }
            Style::Stroke(width) => {
                canvas.stroke_path(&command.path, width, transform, command.color);
            }
        }
    }
}
