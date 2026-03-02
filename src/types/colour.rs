#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Colour {
    White = 0,
    Black = 1,
}
impl Colour {
    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            Colour::White => Colour::Black,
            Colour::Black => Colour::White
        }
    }
    /// Maps colour to its evaluation sign, sign(White) = 1 and sign(Black) = -1.
    #[inline]
    pub const fn sign(self) -> i32 {
        1 - 2 * (self as i32)
    }
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
    /// For FEN parsing
    #[inline]
    pub const fn from_char(c: char) -> Option<Self> {
        match c {
            'w' => Some(Colour::White),
            'b' => Some(Colour::Black),
            _ => None,
        }
    }
    /// Pawn move direction: +8 for White, -8 for Black
    #[inline]
    pub const fn pawn_push(self) -> i8 {
        match self {
            Colour::White => 8,
            Colour::Black => -8,
        }
    }
}

impl std::ops::Not for Colour {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        self.flip()
    }
}