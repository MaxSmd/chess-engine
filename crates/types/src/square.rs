#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Square(u8);

impl Square {
    #[inline]
    pub const fn new(index: u8) -> Option<Self> {
        if index < 64 { Some(Self(index)) } else { None }
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}


