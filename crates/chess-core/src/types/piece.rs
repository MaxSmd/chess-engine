use crate::Colour;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    pub const COUNT: usize = 6;

    pub const PROMOTIONS: [PieceType; 4] = [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen
    ];

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn is_slider(self) -> bool {
        matches!(self, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
    }

    #[inline]
    pub const fn from_char(c: char) -> PieceType {
        match c {
            'P' | 'p' => Some(PieceType::Pawn),
            'N' | 'n' => Some(PieceType::Knight),
            'B' | 'b' => Some(PieceType::Bishop),
            'R' | 'r' => Some(PieceType::Rook),
            'Q' | 'q' => Some(PieceType::Queen),
            'K' | 'k' => Some(PieceType::King),
            _ => None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub colour: Colour,
    pub piece_type: PieceType,
}

impl Piece {
    #[inline]
    pub const fn new(colour: Colour, piece_type: PieceType) -> Self {
        Self { colour, piece_type }
    }

    /// Returns a unique index 0-11 for this piece.
    /// Layout: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
    #[inline]
    pub const fn index(self) -> usize {
        self.colour.index() * 6 + self.piece_type.index()
    }
}