use crate::types::{Colour, Piece};
use crate::board::{Bitboard};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pieces: [Bitboard; 12], // Piece placement: [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
    occupancy: [Bitboard; 3], // Occupancy: [White, Black, Both]

    side_to_move: Colour,
    castling: CastlinRights,
    en_passant: Option<Square>,
    half_move_number: u8,
    full_move_number: u16,

    hash: ZobristHash
}

impl Position {
    pub const fn empty() -> Self {
        Position {
            pieces: [Bitboard::EMPTY; 12],
            occupancy: [Bitboard::EMPTY; 3],

            side_to_move: Colour::WHITE,
            castling: CastlingRights::NONE,
            en_passant: None,
            half_move_number: 0,
            full_move_number: 0,

            hash: Zobrist::ZERO,
        }
    }
}