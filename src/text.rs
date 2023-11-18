use rustybuzz::Face;

pub struct Font {
    pub(crate) face: Face<'static>,
}

impl Font {
    pub fn from_bytes(data: &'static [u8], index: usize) -> Option<Font> {
        let face = Face::from_slice(data, index as u32)?;

        Some(Self { face })
    }
}
