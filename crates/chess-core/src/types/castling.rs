use crate::Colour;

/// Castling rights as 4 bits: WK=1, WQ=2, BK=4, BQ=8
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CastlingRights(u8);


impl CastlingRights {
    pub const NONE: Self = CastlingRights(0);

}