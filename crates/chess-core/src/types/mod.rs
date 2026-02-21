mod colour;
mod piece;
mod square;
mod bitboard;
mod moves;
mod castling;
mod score;

// 2. Re-export public items — controls what users of this module see
pub use colour::Colour;
pub use piece::{Piece, PieceType};
pub use square::Square;
pub use bitboard::Bitboard;
pub use moves::{Move, MoveList, MAX_MOVES};
pub use castling::CastlingRights;
pub use score::Score;