/// Zobrist hash for position identification
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ZobristHash(pub u64);

impl ZobristHash {
    pub const ZERO: Self = ZobristHash(0);
}