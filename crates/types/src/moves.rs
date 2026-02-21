use crate::{Square};

// Bits 0-5: from, Bits 6-11: to, Bits 12-15: flags
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move(u16);

impl Move {
    pub const NULL: Move = Move(0);

    const FLAG_QUIET: u8 = 0;
    const FLAG_DOUBLE_PAWN: u8 = 1;
    const FLAG_KING_CASTLE: u8 = 2;
    const FLAG_QUEEN_CASTLE: u8 = 3;
    const FLAG_CAPTURE: u8 = 4;
    const FLAG_EN_PASSANT: u8 = 5;
    const FLAG_PROMO_N: u8 = 8;
    const FLAG_PROMO_B: u8 = 9;
    const FLAG_PROMO_R: u8 = 10;
    const FLAG_PROMO_Q: u8 = 11;
    const FLAG_PROMO_CAPTURE_N: u8 = 12;
    const FLAG_PROMO_CAPTURE_B: u8 = 13;
    const FLAG_PROMO_CAPTURE_R: u8 = 14;
    const FLAG_PROMO_CAPTURE_Q: u8 = 15;

    pub const fn encode(from: Square, to: Square, flag: u8) -> Self {
        Self(from.index() as u16 | ((to.index() as u16) << 6) | ((flag as u16) << 12))
    }
}

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize
}

impl MoveList {
    #[inline]
    pub const fn new() -> Self {
        Self {moves: Move::Null, MAX_MOVES, len: 0}
    }

    #[inline]
    pub fn push(&mut self, mv: Move) {
        self.moves[self.len] = mv;
        self.len += 1
    }

    #[inline] pub const fn len(&self) -> usize { self.len }
    #[inline] pub const fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn clear(&mut self) { self.len = 0; }
    #[inline] pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().copied()
    }
    #[inline] pub fn contains(&self, mv: Move) -> bool { self.iter().any(|m| m == mv) }
}

impl Default for MoveList {
    fn default() -> Self { Self::new() }
}