use super::Square;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);

    pub const FILE_A: Self = Self(0x0101_0101_0101_0101);
    pub const FILE_B: Self = Self(0x0202_0202_0202_0202);
    pub const FILE_D: Self = Self(0x0808_0808_0808_0808);
    pub const FILE_F: Self = Self(0x2020_2020_2020_2020);
    pub const FILE_E: Self = Self(0x1010_1010_1010_1010);
    pub const FILE_H: Self = Self(0x8080_8080_8080_8080);
    pub const FILE_G: Self = Self(0x4040_4040_4040_4040);
    pub const FILE_C: Self = Self(0x0404_0404_0404_0404);

    pub const RANK_1: Bitboard = Bitboard(0x00000000000000FF);
    pub const RANK_2: Bitboard = Bitboard(0x000000000000FF00);
    pub const RANK_3: Bitboard = Bitboard(0x0000000000FF0000);
    pub const RANK_4: Bitboard = Bitboard(0x00000000FF000000);
    pub const RANK_5: Bitboard = Bitboard(0x000000FF00000000);
    pub const RANK_6: Bitboard = Bitboard(0x0000FF0000000000);
    pub const RANK_7: Bitboard = Bitboard(0x00FF000000000000);
    pub const RANK_8: Bitboard = Bitboard(0xFF00000000000000);

    pub const NOT_FILE_A: Bitboard = Bitboard(!Self::FILE_A.0);
    pub const NOT_FILE_H: Bitboard = Bitboard(!Self::FILE_H.0);

    #[inline]
    pub const fn new(val: u64) -> Self {
        Self(val)
    }

    #[inline]
    pub const fn from_square(sq: Square) -> Self {
        Self(1u64 << sq.index())
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }
}