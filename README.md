# Chess Engine Implementation Plan
## High-Performance Rust Engine — Functional-First, Performance-Driven

**Author:** MaxSmd
**Created:** 2026-02-03
**Version:** 4.0 (Consolidated)
**Target Timeline:** 2–3 weeks (MVP) → 5+ weeks (Production)
**Target Strength:** 1500 ELO (MVP) → 2400+ ELO (Full Implementation)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Technical Design](#4-core-technical-design)
5. [Functional Programming Patterns](#5-functional-programming-patterns)
6. [Performance Engineering](#6-performance-engineering)
7. [Modular Component System](#7-modular-component-system)
8. [Repository Structure](#8-repository-structure)
9. [Implementation Phases](#9-implementation-phases)
   - [Phase 0: Project Bootstrap](#phase-0-project-bootstrap-day-0)
   - [Phase 1: MVP](#phase-1-minimum-viable-product-days-1-14)
   - [Phase 2: Search Enhancements](#phase-2-search-enhancements-days-15-21)
   - [Phase 3: Evaluation Depth](#phase-3-evaluation-depth-days-22-28)
   - [Phase 4: Performance & Polish](#phase-4-performance--polish-days-29-35)
   - [Phase 5: Advanced Extensions](#phase-5-advanced-extensions-days-36)
10. [Technical Specifications](#10-technical-specifications)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Targets](#12-performance-targets)
13. [Development Workflow](#13-development-workflow)
14. [Risk Management](#14-risk-management)
15. [Quick Reference](#15-quick-reference)
16. [Appendices](#16-appendices)

---

## 1. Executive Summary

This document is the single authoritative plan for building a high-performance chess engine in Rust. The architecture prioritises:

- **Performance:** Bitboards, magic move generation, cache-efficient data structures, zero-cost abstractions
- **Functional Style:** Immutable data, pure functions, composable operations, iterator chains
- **Modularity:** Trait-based abstractions allow swapping components without rewrites
- **Extensibility:** Plugin architecture supports NNUE, tablebases, opening books
- **Testability:** Pure functions and clear module boundaries enable isolated unit testing

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Board Representation | Bitboards | O(1) move generation with magic bitboards |
| Move Encoding | `u16` | Compact, cache-friendly |
| Make/Unmake Strategy | Copy-make (functional) | Immutable, thread-safe, simpler reasoning |
| Search Algorithm | Negamax + Alpha-Beta | Industry standard, well-understood |
| Evaluation (MVP) | Material + PST | Fast, reasonable strength |
| Evaluation (Advanced) | NNUE-ready | Architecture supports neural eval |
| Programming Style | Functional-first | Pure functions, immutability, composition |
| Mutation Strategy | Interior mutability only for caches | TT, history tables use `Cell`/`AtomicU64` |

### Timeline Overview

```
Week 1-2:   MVP (playable engine, ~1500 ELO)
Week 3:     Search enhancements (~2000 ELO)
Week 4:     Evaluation improvements (~2200 ELO)
Week 5:     Performance & SMP (~2300 ELO)
Week 6+:    Extensions (NNUE, tablebases) (2400+ ELO)
```

---

## 2. Design Philosophy

### 2.1 Functional-First Principles

This engine follows functional programming principles wherever they don't conflict with performance:

| Principle | Application | Exception |
|-----------|-------------|-----------|
| **Immutability** | Position is immutable, `make_move()` returns new Position | TT, history tables use interior mutability |
| **Pure Functions** | Evaluation, move generation are pure | Search has controlled side effects (TT writes) |
| **No Hidden State** | All inputs explicit, no global mutable state | Lazy-initialised lookup tables (computed once) |
| **Composition** | Build complex behaviour from simple functions | — |
| **Expression-Oriented** | Prefer expressions over statements | — |
| **Iterator Chains** | Use iterators over manual loops | Hot loops may use `for` for performance |

#### Why Functional in a Chess Engine?

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUNCTIONAL BENEFITS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CORRECTNESS                                                 │
│     └── Pure functions are easier to test and reason about     │
│     └── No spooky action at a distance                         │
│     └── Position can't be corrupted by forgotten unmake()      │
│                                                                 │
│  2. PARALLELISM                                                 │
│     └── Immutable positions are inherently thread-safe         │
│     └── No locks needed for position sharing                   │
│     └── Easy Lazy SMP implementation                           │
│                                                                 │
│  3. DEBUGGING                                                   │
│     └── Reproducible: same input = same output                 │
│     └── Can snapshot any position trivially                    │
│     └── No temporal coupling bugs                              │
│                                                                 │
│  4. COMPOSABILITY                                               │
│     └── Evaluators compose: material + pst + mobility          │
│     └── Move ordering composes: hash > captures > killers      │
│     └── Search enhancements layer cleanly                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Performance Without Compromise

Functional style must not sacrifice performance. Key techniques:

| Technique | Description |
|-----------|-------------|
| **Copy-make** | Clone position per node — enables immutability with no overhead vs unmake bugs |
| **Stack allocation** | `MoveList` is `[Move; 256]` on stack, zero heap |
| **Zero-cost iterators** | Iterator chains compile to tight loops |
| **Inline everything hot** | `#[inline(always)]` on critical paths |
| **Const evaluation** | Lookup tables computed at compile time |
| **SIMD-friendly layout** | Data arranged for vectorisation |

### 2.3 Controlled Side Effects

Some side effects are necessary for performance. They are explicitly isolated:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIDE EFFECT BOUNDARIES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PURE (No side effects)                                         │
│  ├── Position::make_move() -> Position                          │
│  ├── evaluate(pos) -> Score                                     │
│  ├── generate_moves(pos) -> MoveList                            │
│  ├── see(pos, move) -> Score                                    │
│  └── is_repetition(pos, history) -> bool                        │
│                                                                 │
│  CONTROLLED MUTATION (Interior mutability, thread-safe)         │
│  ├── TranspositionTable::store()    // AtomicU64 entries        │
│  ├── HistoryTable::update()         // AtomicI16 scores         │
│  └── KillerTable::store()           // Cell<Move>               │
│                                                                 │
│  I/O (Isolated at boundaries)                                   │
│  ├── UCI input parsing                                          │
│  ├── UCI output formatting                                      │
│  └── Opening book / tablebase reads                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UCI Interface                            │
│                    (stdin/stdout — I/O boundary)                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Engine                                 │
│              (orchestrates search, manages state)               │
│         [Only component with controlled mutation]               │
└───────┬─────────────────────┬───────────────────┬───────────────┘
        │                     │                   │
        ▼                     ▼                   ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│    Search     │    │     Eval      │    │     Time      │
│  (pure core,  │    │    (pure)     │    │    (pure)     │
│  TT mutation) │    │               │    │               │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────┐
│   MoveGen     │
│    (pure)     │
└───────┬───────┘
        │
        ▼
┌───────────────┐    ┌───────────────┐
│    Board      │◄───│    Types      │
│(pure/immut.)  │    │ (pure/const)  │
└───────────────┘    └───────────────┘
```

### 3.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     FUNCTIONAL DATA FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FEN String                                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐     parse_fen is pure                              │
│  │  parse  │────────────────────────► Position (immutable)      │
│  └─────────┘                                │                   │
│                           ┌─────────────────┼───────────────┐   │
│                           │                 │               │   │
│                           ▼                 ▼               │   │
│                    ┌──────────┐      ┌──────────┐           │   │
│                    │ generate │      │ evaluate │           │   │
│                    │  moves   │      │          │           │   │
│                    └────┬─────┘      └────┬─────┘           │   │
│                         │                 │                 │   │
│                         ▼                 ▼                 │   │
│                    MoveList            Score                │   │
│                         │                 │                 │   │
│                         └───────┬─────────┘                 │   │
│                                 │                           │   │
│                                 ▼                           │   │
│                          ┌──────────┐                       │   │
│                          │  search  │◄── TT (controlled mut)│   │
│                          └────┬─────┘                       │   │
│                               │                             │   │
│                               ▼                             │   │
│                         SearchResult → Best Move            │   │
│                                                             │   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Module Dependency Graph

```
                    ┌──────────────┐
                    │     uci      │  ◄── I/O boundary
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    engine    │  ◄── Orchestrator (controlled mutation)
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  search  │      │   eval   │      │   time   │
  │  (TT mut)│      │  (pure)  │      │  (pure)  │
  └────┬─────┘      └────┬─────┘      └──────────┘
       │                 │
       │     ┌───────────┤
       ▼     ▼           ▼
  ┌──────────────┐  ┌──────────┐
  │   movegen    │  │  tables  │  ◄── const / lazy_static
  │    (pure)    │  │  (pure)  │
  └──────┬───────┘  └──────────┘
         │
         ▼
  ┌──────────────┐
  │    board     │  ◄── Immutable Position
  └──────────────┘
         │
         ▼
  ┌──────────────┐
  │    types     │  ◄── Pure data types, const fns
  └──────────────┘
```

**Rule: Dependencies only flow downward. No cycles.**

### 3.4 Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAIT HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Position   │    │  MoveGen    │    │     Evaluator       │  │
│  │   Trait     │◄───│   Trait     │    │       Trait         │  │
│  │(pure/immut) │    │   (pure)    │    │      (pure)         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        ▲                  ▲                      ▲              │
│        │                  │                      │              │
│  ┌─────┴─────┐     ┌──────┴──────┐     ┌─────────┴─────────┐    │
│  │ Bitboard  │     │   Magic     │     │  Material │ NNUE  │    │
│  │ Position  │     │   MoveGen   │     │  Evaluator│ Eval  │    │
│  └───────────┘     └─────────────┘     └───────────┴───────┘    │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Searcher   │    │  TimeCtrl   │    │   TransTable        │  │
│  │   Trait     │    │   Trait     │    │      Trait          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│        ▲                  ▲                      ▲              │
│        │                  │                      │              │
│  ┌─────┴─────┐     ┌──────┴──────┐     ┌─────────┴─────────┐    │
│  │ AlphaBeta │     │  Standard   │     │  HashMap  │ Lock  │    │
│  │   MCTS    │     │  Sudden     │     │    TT     │ Free  │    │
│  └───────────┘     └─────────────┘     └───────────┴───────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Technical Design

### 4.1 Board Representation — Immutable Bitboards

Position is immutable. All operations return new values:

```rust
// Immutable position — all fields private, no mutation methods
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Position {
    // Hot data first (accessed every node)
    pieces: [Bitboard; 12],      // 96 bytes  — one per piece type per color
    occupancy: [Bitboard; 3],    // 24 bytes  — White, Black, All
    hash: ZobristHash,           // 8 bytes

    // Cold data
    side_to_move: Color,         // 1 byte
    castling: CastlingRights,    // 1 byte    — 4 bits used
    en_passant: Option<Square>,  // 2 bytes
    halfmove_clock: u8,          // 1 byte
    fullmove_number: u16,        // 2 bytes
    _padding: u8,                // 1 byte    — alignment
}  // Total: 136 bytes — fits in 2–3 cache lines

impl Position {
    /// Pure function — returns new position, self unchanged
    #[inline]
    pub fn make_move(self, mv: Move) -> Self {
        let mut new = self;
        let from = mv.from();
        let to = mv.to();
        let piece = self.piece_at(from).unwrap();

        // Update bitboards — pure transformations
        new.pieces[piece.index()] =
            self.pieces[piece.index()].without(from).with(to);
        new.occupancy[self.side_to_move as usize] =
            self.occupancy[self.side_to_move as usize].without(from).with(to);

        // Handle captures, castling, en passant — all pure
        new.side_to_move = self.side_to_move.flip();
        new.hash = self.hash.update(/* incremental */);
        new
    }

    #[inline(always)]
    pub const fn side_to_move(&self) -> Color { self.side_to_move }

    #[inline(always)]
    pub const fn piece_at(&self, sq: Square) -> Option<Piece> { /* ... */ }
}
```

### 4.2 Zobrist Hashing — Compile-Time

```rust
// Zero runtime cost — computed at compile time
pub const ZOBRIST_KEYS: ZobristKeys = ZobristKeys::generate();

pub struct ZobristKeys {
    pieces: [[[u64; 64]; 6]; 2],  // [color][piece][square]
    castling: [u64; 16],
    en_passant: [u64; 8],
    side_to_move: u64,
}

impl ZobristKeys {
    pub const fn generate() -> Self {
        // Compile-time PRNG (xorshift or similar)
        // ...
    }
}
```

### 4.3 Move Generation — Pure Functions

#### Magic Bitboards (Const Tables)

```rust
// All tables computed at compile time
pub static BISHOP_MAGICS: [MagicEntry; 64] = generate_bishop_magics();
pub static ROOK_MAGICS: [MagicEntry; 64] = generate_rook_magics();

// Pure function for attack generation
#[inline(always)]
pub const fn bishop_attacks(sq: Square, occupied: Bitboard) -> Bitboard {
    let entry = &BISHOP_MAGICS[sq.0 as usize];
    let index = ((occupied.0 & entry.mask) * entry.magic) >> entry.shift;
    Bitboard(BISHOP_ATTACKS[entry.offset + index as usize])
}
```

#### Move Encoding (`u16`)

```
bits 0-5:   from square (0-63)
bits 6-11:  to square (0-63)
bits 12-15: flag
  0000: quiet move           0001: double pawn push
  0010: king castle          0011: queen castle
  0100: capture              0101: en passant
  1000-1011: promotions (N, B, R, Q)
  1100-1111: promotion captures
```

#### Functional Move Generation

```rust
/// Pure move generation — no side effects
#[inline]
pub fn generate_moves(pos: &Position) -> MoveList {
    if pos.is_in_check() {
        generate_evasions(pos)
    } else {
        generate_all(pos)
    }
}

/// Iterator-based legal move filtering
pub fn legal_moves(pos: &Position) -> impl Iterator<Item = Move> + '_ {
    generate_moves(pos)
        .into_iter()
        .filter(move |&mv| pos.make_move(mv).is_legal())
}

/// Pure perft
pub fn perft(pos: &Position, depth: u8) -> u64 {
    if depth == 0 { return 1; }
    legal_moves(pos)
        .map(|mv| perft(&pos.make_move(mv), depth - 1))
        .sum()
}
```

#### Generation Order

1. If in check → generate evasions only
2. Otherwise: king moves (castling) → knight → pawn (pushes, captures, EP, promotions) → sliding pieces (magic lookups)
3. Filter illegal moves (leaves king in check)

### 4.4 Evaluation — Pure, Composable

#### MVP Evaluation

```rust
// Pure evaluation function — no side effects
#[inline]
pub fn evaluate(pos: &Position) -> Score {
    if is_material_draw(pos) { return Score::DRAW; }

    let phase = game_phase(pos);

    let mg_score = material_score(pos, Phase::Middlegame)
        + pst_score(pos, Phase::Middlegame);

    let eg_score = material_score(pos, Phase::Endgame)
        + pst_score(pos, Phase::Endgame);

    // Tapered eval — pure interpolation
    taper(mg_score, eg_score, phase)
}
```

#### Composable Evaluators

```rust
// Compose evaluators functionally
pub fn weighted_eval(
    evaluators: &[(fn(&Position) -> Score, i32)]
) -> impl Fn(&Position) -> Score + '_ {
    move |pos| {
        evaluators.iter()
            .map(|(f, weight)| f(pos) * *weight / 100)
            .sum()
    }
}

// Usage
let eval = weighted_eval(&[
    (material_score, 100),
    (pst_score, 100),
    (mobility_score, 50),
    (king_safety_score, 75),
]);
```

#### Piece Values

| Piece | Middlegame | Endgame |
|-------|------------|---------|
| Pawn | 100 | 120 |
| Knight | 320 | 300 |
| Bishop | 330 | 320 |
| Rook | 500 | 550 |
| Queen | 900 | 950 |

#### Advanced Evaluation Components (Phases 3–5)

| Component | Description | Complexity |
|-----------|-------------|------------|
| Pawn Structure | Doubled, isolated, passed pawns | Medium |
| King Safety | Pawn shield, attacker count | High |
| Mobility | Legal move count per piece | Medium |
| Piece Placement | Outposts, rooks on open files | Low |
| Tapered Eval | Interpolate MG/EG by game phase | Medium |
| Bishop Pair | Bonus for both bishops | Low |

#### Lazy Pawn Hash Table (Controlled Mutation)

```rust
// Thread-local pawn hash — interior mutability, pure interface
thread_local! {
    static PAWN_HASH: RefCell<PawnHashTable> = RefCell::new(PawnHashTable::new());
}

pub fn pawn_structure_score(pos: &Position) -> Score {
    PAWN_HASH.with(|table| {
        let mut table = table.borrow_mut();
        table.probe_or_compute(pos.pawn_hash(), || {
            compute_pawn_structure(pos)  // Pure computation
        })
    })
}
```

### 4.5 Search Algorithm — Functional Core

#### Negamax with Alpha-Beta

```rust
// Search context — passed explicitly, no global state
pub struct SearchContext<'a> {
    tt: &'a TranspositionTable,      // Interior mutability
    history: &'a HistoryTable,       // Interior mutability
    killers: &'a KillerTable,        // Interior mutability
    limits: SearchLimits,
    nodes: AtomicU64,
    stop: AtomicBool,
}

// Iterative deepening — functional fold
fn iterative_deepening(pos: &Position, ctx: &SearchContext) -> SearchResult {
    (1..=MAX_DEPTH)
        .take_while(|_| !ctx.should_stop())
        .fold(SearchResult::default(), |best, depth| {
            let result = aspiration_search(pos, depth, best.score, ctx);
            if result.score > best.score { result } else { best }
        })
}

// Negamax — functional recursion with try_fold for beta cutoff
fn negamax(
    pos: &Position, depth: u8, mut alpha: Score, beta: Score,
    ctx: &SearchContext, ply: u8,
) -> Score {
    if depth == 0 { return quiescence(pos, alpha, beta, ctx); }

    // TT probe (controlled side effect)
    if let Some(entry) = ctx.tt.probe(pos.hash()) {
        if entry.depth >= depth && entry.can_use(alpha, beta) {
            return entry.score;
        }
    }

    // Generate and order moves (pure)
    let moves = generate_moves(pos);
    let ordered = order_moves(pos, &moves, ctx, ply);

    // Search with try_fold for early exit on beta cutoff
    let init = SearchState { alpha, best: Score::NEG_INF, best_move: None };
    let result = ordered
        .take_while(|_| !ctx.should_stop())
        .try_fold(init, |mut state, mv| {
            let new_pos = pos.make_move(mv);  // Immutable!
            let score = -negamax(&new_pos, depth - 1, -beta, -state.alpha, ctx, ply + 1);

            if score > state.best {
                state.best = score;
                state.best_move = Some(mv);
                state.alpha = state.alpha.max(score);
            }

            if score >= beta {
                ctx.killers.store(ply, mv);   // Controlled mutation
                ctx.history.update(mv, depth); // Controlled mutation
                ControlFlow::Break(state)
            } else {
                ControlFlow::Continue(state)
            }
        });

    let state = match result {
        ControlFlow::Break(s) | ControlFlow::Continue(s) => s,
    };

    // TT store (controlled side effect)
    ctx.tt.store(pos.hash(), TTEntry::new(state.best, state.best_move, depth));
    state.best
}
```

#### Search Enhancements (Priority Order)

| Technique | Description | ELO Gain |
|-----------|-------------|----------|
| Iterative Deepening | Search depth 1, 2, 3... until time up | Foundation |
| Transposition Table | Cache positions, avoid re-search | +100 |
| MVV-LVA Ordering | Capture most valuable with least valuable | +50 |
| Killer Moves | Remember quiet moves that caused cutoffs | +40 |
| History Heuristic | Score moves by historical success | +30 |
| PVS | Null-window search after first move | +50 |
| Null Move Pruning | Skip turn to prove position is good | +80 |
| Late Move Reductions | Reduce depth for likely-bad moves | +100 |
| Quiescence Search | Extend captures to avoid horizon effect | Essential |
| SEE | Prune losing captures | +50 |
| Futility Pruning | Skip hopeless moves at low depth | +40 |
| Aspiration Windows | Narrow alpha-beta window around expected | +20 |

#### Move Ordering Strategy

```
1. Hash move (from transposition table)
2. Winning captures (MVV-LVA or SEE > 0)
3. Killer moves (2 slots per ply)
4. Counter moves
5. History heuristic (quiet moves by success rate)
6. Losing captures (SEE < 0)
```

#### Functional Move Ordering

```rust
pub fn order_moves<'a>(
    pos: &'a Position, moves: &'a MoveList,
    ctx: &'a SearchContext, ply: u8,
) -> impl Iterator<Item = Move> + 'a {
    let scored: Vec<_> = moves.iter()
        .map(|&mv| (mv, score_move(pos, mv, ctx, ply)))
        .collect();

    scored.into_iter()
        .sorted_by_key(|(_, score)| std::cmp::Reverse(*score))
        .map(|(mv, _)| mv)
}

// Pure move scoring — composable heuristics
#[inline]
fn score_move(pos: &Position, mv: Move, ctx: &SearchContext, ply: u8) -> i32 {
    hash_move_bonus(pos, mv, ctx)
        .or_else(|| capture_score(pos, mv))
        .or_else(|| killer_score(mv, ctx, ply))
        .or_else(|| history_score(mv, ctx))
        .unwrap_or(0)
}

#[inline]
const fn capture_score(pos: &Position, mv: Move) -> Option<i32> {
    let captured = pos.piece_at(mv.to())?;
    let attacker = pos.piece_at(mv.from())?;
    Some(mvv_lva_score(captured, attacker) + 10_000)
}
```

### 4.6 Time Management

#### Basic Strategy

```
allocated_time = remaining_time / estimated_moves_left + increment

estimated_moves_left:
    - Opening (moves < 10): 40
    - Middlegame: 30
    - Endgame (few pieces): 20
```

#### Advanced Features

- Extend time if best move is unstable
- Use less time if position is clearly winning/losing
- Respect move overhead for network latency

---

## 5. Functional Programming Patterns

### 5.1 Expression-Oriented Style

```rust
// AVOID: Statement-oriented
fn evaluate_bad(pos: &Position) -> Score {
    let mut score = Score::ZERO;
    score = score + material(pos);
    score = score + pst(pos);
    if pos.is_endgame() {
        score = score + endgame_bonus(pos);
    }
    return score;
}

// PREFER: Expression-oriented
fn evaluate(pos: &Position) -> Score {
    material(pos) + pst(pos)
        + if pos.is_endgame() { endgame_bonus(pos) } else { Score::ZERO }
}

// OR: Using combinators
fn evaluate(pos: &Position) -> Score {
    [material, pst, mobility, king_safety].iter().map(|f| f(pos)).sum()
}
```

### 5.2 Iterator Chains Over Loops

```rust
// AVOID: Manual loop
fn count_pieces_bad(pos: &Position, color: Color) -> u32 {
    let mut count = 0;
    for piece_type in PieceType::ALL { count += pos.pieces(color, piece_type).count_ones(); }
    count
}

// PREFER: Iterator chain
fn count_pieces(pos: &Position, color: Color) -> u32 {
    PieceType::ALL.iter()
        .map(|&pt| pos.pieces(color, pt).count_ones())
        .sum()
}
```

### 5.3 Option/Result Combinators

```rust
// AVOID: Nested if-lets
fn get_capture_score_bad(pos: &Position, mv: Move) -> Option<Score> {
    if let Some(captured) = pos.piece_at(mv.to()) {
        if let Some(attacker) = pos.piece_at(mv.from()) {
            return Some(mvv_lva(captured, attacker));
        }
    }
    None
}

// PREFER: Combinator chain
fn get_capture_score(pos: &Position, mv: Move) -> Option<Score> {
    pos.piece_at(mv.to())
        .zip(pos.piece_at(mv.from()))
        .map(|(captured, attacker)| mvv_lva(captured, attacker))
}
```

### 5.4 Const Functions for Compile-Time Computation

```rust
pub const fn init_king_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard(0); 64];
    let mut sq = 0;
    while sq < 64 {
        attacks[sq] = compute_king_attacks(Square(sq as u8));
        sq += 1;
    }
    attacks
}

pub static KING_ATTACKS: [Bitboard; 64] = init_king_attacks();  // Zero runtime cost

#[inline(always)]
pub const fn king_attacks(sq: Square) -> Bitboard { KING_ATTACKS[sq.0 as usize] }
```

### 5.5 Type-Safe Builder Pattern

```rust
pub struct EngineBuilder<E = (), M = (), S = ()> {
    evaluator: E, movegen: M, searcher: S, tt_size: usize, threads: usize,
}

impl<E, M, S> EngineBuilder<E, M, S> {
    // Each method returns a new builder (functional transformation)
    pub fn with_evaluator<E2>(self, eval: E2) -> EngineBuilder<E2, M, S> {
        EngineBuilder { evaluator: eval, movegen: self.movegen, searcher: self.searcher,
                        tt_size: self.tt_size, threads: self.threads }
    }
    pub fn with_tt_size(self, mb: usize) -> Self { Self { tt_size: mb, ..self } }
}
```

### 5.6 Newtype Pattern for Type Safety

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(u8);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score(i16);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZobristHash(u64);

// Compiler prevents mixing up Square and Score at zero cost
```

### 5.7 Try-Fold for Early Search Exit

```rust
use std::ops::ControlFlow;

fn search_with_cutoff(
    pos: &Position, moves: impl Iterator<Item = Move>,
    ctx: &SearchContext, depth: u8, alpha: Score, beta: Score,
) -> SearchResult {
    let init = SearchState { alpha, best: Score::NEG_INF, best_move: None };

    let result = moves.try_fold(init, |mut state, mv| {
        let score = -negamax(&pos.make_move(mv), depth - 1, -beta, -state.alpha, ctx);
        if score > state.best {
            state.best = score;
            state.best_move = Some(mv);
            state.alpha = state.alpha.max(score);
        }
        if score >= beta { ControlFlow::Break(state) }
        else { ControlFlow::Continue(state) }
    });

    match result { ControlFlow::Break(s) | ControlFlow::Continue(s) => s.into() }
}
```

---

## 6. Performance Engineering

### 6.1 Zero-Cost Abstractions

| Abstraction | Cost | How Achieved |
|-------------|------|--------------|
| Iterators | Zero | Compile to same code as manual loops |
| Newtypes | Zero | Same representation as underlying type |
| Trait bounds | Zero | Monomorphisation |
| `Option<NonZeroU8>` | Zero | Null pointer optimisation |
| Closures | Usually zero | Inlined by compiler |

### 6.2 Cache Optimisation

```rust
// TT Entry optimised for cache
#[repr(C, align(16))]
pub struct TTEntry {
    hash_verify: u32,    // 4 bytes
    score: i16,          // 2 bytes
    best_move: Move,     // 2 bytes
    depth: u8,           // 1 byte
    flag: TTFlag,        // 1 byte
    age: u8,             // 1 byte
    _padding: [u8; 5],   // 5 bytes
}  // Total: 16 bytes — cache-aligned

// Prefetch for TT
#[inline(always)]
pub fn prefetch_tt(&self, hash: ZobristHash) {
    let index = self.index(hash);
    unsafe {
        std::arch::x86_64::_mm_prefetch(
            self.entries.as_ptr().add(index) as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        );
    }
}
```

### 6.3 SIMD Opportunities

```rust
// SIMD-friendly evaluation accumulator (for NNUE path)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[repr(align(32))]
pub struct Accumulator {
    values: [i16; 256],  // 512 bytes, 32-byte aligned
}

impl Accumulator {
    #[target_feature(enable = "avx2")]
    unsafe fn add_feature(&mut self, feature: usize, weights: &[i16; 256]) {
        for i in (0..256).step_by(16) {
            let acc = _mm256_load_si256(self.values[i..].as_ptr() as *const __m256i);
            let wgt = _mm256_load_si256(weights[i..].as_ptr() as *const __m256i);
            let sum = _mm256_add_epi16(acc, wgt);
            _mm256_store_si256(self.values[i..].as_mut_ptr() as *mut __m256i, sum);
        }
    }
}
```

### 6.4 Inlining Strategy

```rust
#[inline(always)]  // Hot paths — always inline
pub fn make_move(self, mv: Move) -> Self { /* ... */ }
pub fn piece_at(&self, sq: Square) -> Option<Piece> { /* ... */ }
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard { /* ... */ }

#[inline(never)] #[cold]  // Cold paths — never inline
fn report_illegal_position(pos: &Position) { /* ... */ }

#[inline]  // Medium paths — let compiler decide
pub fn evaluate(&self, pos: &Position) -> Score { /* ... */ }
```

### 6.5 Memory Layout

```rust
// Move: packed into 16 bits
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Move(u16);

impl Move {
    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        Self(from.0 as u16 | ((to.0 as u16) << 6) | ((flag as u16) << 12))
    }
    #[inline(always)]
    pub const fn from(self) -> Square { Square((self.0 & 0x3F) as u8) }
    #[inline(always)]
    pub const fn to(self) -> Square { Square(((self.0 >> 6) & 0x3F) as u8) }
}

// MoveList: stack-allocated, no heap
pub struct MoveList {
    moves: [Move; 256],
    len: u8,
}
```

### 6.6 Branch Reduction

```rust
// AVOID: Branches in hot code
fn piece_value_bad(piece: PieceType) -> Score {
    match piece { PieceType::Pawn => Score(100), /* ... */ }
}

// PREFER: Lookup table
static PIECE_VALUES: [Score; 6] = [Score(100), Score(320), Score(330), Score(500), Score(900), Score(0)];

#[inline(always)]
fn piece_value(piece: PieceType) -> Score { PIECE_VALUES[piece as usize] }
```

### 6.7 Branch Prediction Hints

```rust
#[inline(always)]
fn likely(b: bool) -> bool { if !b { std::hint::cold() } b }

#[inline(always)]
fn unlikely(b: bool) -> bool { if b { std::hint::cold() } b }
```

---

## 7. Modular Component System

### 7.1 Core Traits

#### Position Trait

```rust
/// Immutable position interface
pub trait Position: Clone + Copy + Eq + Hash + Send + Sync {
    fn piece_at(&self, sq: Square) -> Option<Piece>;
    fn pieces(&self, color: Color, pt: PieceType) -> Bitboard;
    fn side_to_move(&self) -> Color;
    fn castling_rights(&self) -> CastlingRights;
    fn en_passant_square(&self) -> Option<Square>;
    fn halfmove_clock(&self) -> u8;
    fn hash(&self) -> ZobristHash;

    /// Pure transformation — returns new position
    fn make_move(&self, mv: Move) -> Self;

    fn is_check(&self) -> bool;
    fn is_legal(&self) -> bool;
}
```

#### MoveGenerator Trait

```rust
/// Pure move generation interface
pub trait MoveGenerator<P: Position>: Send + Sync {
    fn generate_all(&self, pos: &P) -> MoveList;
    fn generate_captures(&self, pos: &P) -> MoveList;
    fn generate_quiet(&self, pos: &P) -> MoveList;
    fn legal_moves<'a>(&'a self, pos: &'a P) -> impl Iterator<Item = Move> + 'a;
    fn perft(&self, pos: &P, depth: u8) -> u64;
}
```

#### Evaluator Trait

```rust
/// Pure evaluation interface
pub trait Evaluator<P: Position>: Send + Sync {
    fn evaluate(&self, pos: &P) -> Score;
    fn evaluate_relative(&self, pos: &P) -> Score {
        let score = self.evaluate(pos);
        if pos.side_to_move() == Color::White { score } else { -score }
    }
    fn trace(&self, pos: &P) -> EvalTrace;
}

/// Compose evaluators
impl<P: Position> Evaluator<P> for Vec<Box<dyn Evaluator<P>>> {
    fn evaluate(&self, pos: &P) -> Score { self.iter().map(|e| e.evaluate(pos)).sum() }
}
```

#### Searcher Trait

```rust
pub trait Searcher<P: Position>: Send {
    fn search(&self, pos: &P, ctx: &SearchContext) -> SearchResult;
    fn stop(&self);
    fn stats(&self) -> SearchStats;
}
```

#### TransTable Trait

```rust
/// Thread-safe transposition table — interior mutability
pub trait TransTable: Send + Sync {
    fn probe(&self, hash: ZobristHash) -> Option<TTEntry>;
    fn store(&self, hash: ZobristHash, entry: TTEntry);  // Controlled mutation
    fn hashfull(&self) -> u16;
    fn clear(&self);
    fn resize(&self, mb: usize);
}
```

#### TimeController Trait

```rust
pub trait TimeController: Send {
    fn start(&mut self, limits: SearchLimits);
    fn should_stop(&self) -> bool;
    fn elapsed(&self) -> Duration;
    fn allocate_time(&self, ply: u16) -> Duration;
}
```

### 7.2 Engine Builder

```rust
let engine = EngineBuilder::new()
    .with_position_type::<BitboardPosition>()
    .with_movegen(MagicMoveGen::new())
    .with_evaluator(
        CompositeEval::new()
            .add(MaterialEval::new(), 1.0)
            .add(PSTEval::load("pst.bin"), 1.0)
            .add(PawnStructureEval::new(), 0.8)
    )
    .with_searcher(AlphaBetaSearcher::new())
    .with_tt(LockFreeTT::new(256))  // 256 MB
    .with_time_controller(StandardTimeControl::new())
    .with_opening_book(PolyglotBook::open("book.bin")?)
    .with_syzygy_path("/path/to/tablebases")
    .with_threads(4)
    .build();
```

### 7.3 Feature Flags

```toml
[features]
default = ["classical-eval", "magic-movegen"]

# Evaluation
classical-eval = []
nnue-eval = ["dep:nnue-rs"]

# Move generation
magic-movegen = []
pext-movegen = []          # Requires BMI2

# Search
smp = ["dep:crossbeam"]
mcts = []

# Extras
syzygy = ["dep:fathom"]
opening-book = []

# Development
tuning = []
trace = []                 # Detailed eval traces
```

### 7.4 Extension Points

```rust
// Adding a custom evaluator
impl Evaluator<BitboardPosition> for MyNNUE {
    fn evaluate(&self, pos: &BitboardPosition) -> Score { /* your impl */ }
}
let engine = EngineBuilder::new().with_evaluator(MyNNUE::load("weights.bin")).build();

// Opening book trait
pub trait OpeningBook {
    fn probe(&self, hash: u64) -> Option<Move>;
    fn learn(&mut self, game: &[Move], result: f32);
}
```

---

## 8. Repository Structure

```
chess-engine/
├── Cargo.toml                      # Workspace root
├── Cargo.lock
├── README.md
├── LICENSE
├── rustfmt.toml                    # Functional-style formatting
├── clippy.toml                     # Pedantic + functional lints
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Build, test, clippy, fmt
│       └── bench.yml               # Performance regression tests
│
├── crates/
│   ├── types/                      # Zero-dependency, pure types
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── piece.rs            # Piece — const fns
│   │       ├── colour.rs           # Colour — const fns
│   │       ├── square.rs           # Square — const fns
│   │       ├── bitboard.rs         # Bitboard ops — const fns
│   │       ├── castling.rs         # CastlingRights — const fns
│   │       ├── moves.rs            # Move encoding — const fns
│   │       └── score.rs            # Score type — const fns
│   │
│   ├── board/                      # Immutable position
│   │   ├── Cargo.toml              # depends on: types
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── position.rs         # Immutable Position
│   │       ├── zobrist.rs          # Const hash keys
│   │       ├── fen.rs              # Pure FEN parsing
│   │       └── makemove.rs         # Pure make_move()
│   │
│   ├── movegen/                    # Pure move generation
│   │   ├── Cargo.toml              # depends on: types, board
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs           # MoveGenerator trait
│   │       ├── magic/
│   │       │   ├── mod.rs
│   │       │   ├── tables.rs       # Const magic tables
│   │       │   └── attacks.rs      # Pure attack functions
│   │       ├── generator.rs        # Pure move generation
│   │       ├── perft.rs            # Pure perft
│   │       └── see.rs              # Pure static exchange eval
│   │
│   ├── eval/                       # Pure evaluation
│   │   ├── Cargo.toml              # depends on: types, board
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs           # Evaluator trait
│   │       ├── classical/
│   │       │   ├── mod.rs
│   │       │   ├── material.rs     # Pure material eval
│   │       │   ├── pst.rs          # Const PST tables
│   │       │   ├── pawns.rs        # Pure + pawn hash cache
│   │       │   ├── king_safety.rs  # Pure king safety
│   │       │   ├── mobility.rs     # Pure mobility
│   │       │   ├── endgame.rs      # Pure endgame eval
│   │       │   └── weights.rs      # Tunable parameters
│   │       ├── nnue/               # Neural network eval (future)
│   │       │   ├── mod.rs
│   │       │   ├── network.rs
│   │       │   └── accumulator.rs
│   │       └── composite.rs        # Evaluator composition
│   │
│   ├── search/                     # Search with controlled mutation
│   │   ├── Cargo.toml              # depends on: types, board, movegen, eval
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs           # Searcher trait
│   │       ├── context.rs          # SearchContext (explicit state)
│   │       ├── alphabeta/
│   │       │   ├── mod.rs
│   │       │   ├── negamax.rs      # Functional recursion
│   │       │   ├── pvs.rs          # PVS variant
│   │       │   ├── quiescence.rs   # Quiescence search
│   │       │   └── pruning.rs      # All pruning techniques
│   │       ├── mcts/               # Monte Carlo (future)
│   │       │   └── mod.rs
│   │       ├── ordering/
│   │       │   ├── mod.rs
│   │       │   ├── mvv_lva.rs      # Pure scoring
│   │       │   ├── killers.rs      # Interior mutability
│   │       │   ├── history.rs      # Interior mutability
│   │       │   └── orderer.rs      # Composite orderer
│   │       ├── tt/
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs       # TransTable trait
│   │       │   ├── table.rs        # Lock-free TT
│   │       │   └── entry.rs        # TTEntry
│   │       └── limits.rs           # SearchLimits (immutable)
│   │
│   ├── time/                       # Pure time management
│   │   ├── Cargo.toml              # depends on: types
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── traits.rs           # TimeController trait
│   │       ├── standard.rs         # Incremental time control
│   │       ├── fixed.rs            # Fixed time/depth
│   │       └── sudden_death.rs     # No increment
│   │
│   ├── engine/                     # Orchestration layer
│   │   ├── Cargo.toml              # depends on: all above
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs           # Engine struct
│   │       ├── builder.rs          # Functional builder
│   │       ├── config.rs           # Immutable config
│   │       └── thread_pool.rs      # SMP management
│   │
│   └── uci/                        # I/O boundary
│       ├── Cargo.toml              # depends on: engine
│       └── src/
│           ├── lib.rs
│           ├── parser.rs           # Pure parsing
│           ├── handler.rs          # Command handling
│           ├── output.rs           # Pure formatting
│           └── options.rs          # UCI options
│
├── bins/
│   └── chess-engine/               # Main binary
│       ├── Cargo.toml
│       └── src/
│           └── main.rs             # I/O entry point
│
├── benches/
│   ├── perft_bench.rs
│   ├── eval_bench.rs
│   └── search_bench.rs
│
├── examples/
│   ├── custom_eval.rs              # How to plug in custom evaluator
│   ├── analysis_mode.rs            # Use as library
│   └── tournament.rs               # Run automated matches
│
├── tests/
│   ├── integration/
│   │   ├── perft_suite.rs
│   │   ├── epd_suite.rs
│   │   └── uci_compliance.rs
│   └── fixtures/
│       └── positions.epd
│
├── resources/
│   ├── epd/
│   │   ├── wac.epd                 # Win at Chess suite
│   │   └── sts.epd                 # Strategic Test Suite
│   └── books/
│       └── opening_book.bin
│
└── docs/
    ├── ARCHITECTURE.md
    ├── FUNCTIONAL_PATTERNS.md
    ├── PERFORMANCE.md
    ├── TUNING.md
    └── CONTRIBUTING.md
```

---

## 9. Implementation Phases

### Phase 0: Project Bootstrap (Day 0)

#### Tasks

- [ ] Create repository structure
- [ ] Set up workspace Cargo.toml
- [ ] Configure rustfmt and clippy for functional style
- [ ] Create README with design philosophy
- [ ] Set up basic CI workflow

#### Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/types",
    "crates/board",
    "crates/movegen",
    "crates/eval",
    "crates/search",
    "crates/engine",
    "crates/uci",
    "bins/chess-engine",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"

[workspace.dependencies]
types = { path = "crates/types" }
board = { path = "crates/board" }
movegen = { path = "crates/movegen" }
eval = { path = "crates/eval" }
search = { path = "crates/search" }
engine = { path = "crates/engine" }
uci = { path = "crates/uci" }
thiserror = "1.0"
rand = "0.8"
```

#### Clippy Configuration

```toml
# clippy.toml
avoid-breaking-exported-api = false
cognitive-complexity-threshold = 15

# .clippy.args
-W clippy::pedantic
-W clippy::nursery
-W clippy::unwrap_used
-W clippy::expect_used
-A clippy::module_name_repetitions
-W clippy::manual_let_else
-W clippy::match_same_arms
-W clippy::redundant_closure_for_method_calls
-W clippy::unnecessary_wraps
```

#### Rustfmt Configuration

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
use_field_init_shorthand = true
use_try_shorthand = true
chain_width = 60
fn_call_width = 80
```

#### Decision Checklist

- [x] Copy-make vs unmake-move? → **Copy-make** (functional, immutable, thread-safe)
- [x] Move encoding: u16 vs u32? → **u16** (compact, cache-friendly)
- [x] Pre-computed vs runtime magics? → **Pre-computed** (const where possible)
- [x] TT entry size? → **16 bytes** (cache-aligned)

---

### Phase 1: Minimum Viable Product (Days 1–14)

**Goal:** A working engine with functional architecture that can play legal chess and beat beginners
**Target Strength:** ~1500 ELO

#### MVP-1: Foundation Layer (Days 1–2)

##### Day 1: Types Crate (Pure, Const)

| File | Contents | Notes |
|------|----------|-------|
| `piece.rs` | `Piece`, `Color`, `PieceType` enums | All const fns |
| `square.rs` | `Square` (0-63), `File`, `Rank` | All const fns |
| `bitboard.rs` | `Bitboard` (u64) with functional iterator | Const where possible |
| `moves.rs` | `Move` (u16), `MoveFlag`, `MoveList` | Stack-allocated list |
| `castling.rs` | `CastlingRights` (4 bits) | Const bit manipulation |
| `score.rs` | `Score` type with operators, `MATE`/`DRAW` constants | Const arithmetic |

**Example: Functional Bitboard**

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const ALL: Self = Self(!0);

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool { (self.0 >> sq.0) & 1 != 0 }

    #[inline(always)]
    pub const fn with(self, sq: Square) -> Self { Self(self.0 | (1 << sq.0)) }

    #[inline(always)]
    pub const fn without(self, sq: Square) -> Self { Self(self.0 & !(1 << sq.0)) }

    pub fn iter_squares(self) -> impl Iterator<Item = Square> {
        std::iter::successors(
            (self.0 != 0).then(|| Square(self.0.trailing_zeros() as u8)),
            move |_| {
                let remaining = self.0 & (self.0 - 1);
                (remaining != 0).then(|| Square(remaining.trailing_zeros() as u8))
            }
        )
    }
}
```

**Deliverable:** All types compile, basic unit tests pass

##### Day 2: Board Crate (Immutable)

| File | Contents | Notes |
|------|----------|-------|
| `position.rs` | Immutable `Position` struct, game state | Copy semantics |
| `zobrist.rs` | Const hash generation, incremental updates | Compile-time tables |
| `fen.rs` | Pure FEN parsing and generation | Returns `Result<Position>` |
| `makemove.rs` | Pure `make_move()` → returns new `Position` | All pure transformations |

**Deliverable:** Can parse FEN, make moves, generate FEN back

#### MVP-2: Move Generation (Days 3–5)

##### Day 3: Attack Tables

| File | Contents | Debugging |
|------|----------|-----------|
| `tables.rs` | Knight, king, pawn attack tables (const) | Low |
| `magic/` | Magic bitboard tables (pre-computed) | Low |
| `attacks.rs` | `bishop_attacks()`, `rook_attacks()` (pure) | Medium |

##### Day 4: Move Generator

| File | Contents | Debugging |
|------|----------|-----------|
| `generator.rs` | `generate_moves()`, `generate_captures()` — all pure | **HIGH** |

##### Day 5: Legal Moves & Perft Validation

| File | Contents | Debugging |
|------|----------|-----------|
| `legality.rs` | Pin detection, check detection | **HIGH** |
| `perft.rs` | Pure perft function for validation | Low |

**⚠️ Critical: Perft Validation — DO NOT PROCEED UNTIL PERFT PASSES**

| Position | Depth | Expected Nodes |
|----------|-------|----------------|
| Startpos | 5 | 4,865,609 |
| Kiwipete | 4 | 4,085,603 |
| Position 3 | 6 | 11,030,083 |
| Position 4 | 5 | 15,833,292 |
| Position 5 | 5 | 89,941,194 |

Common bugs to watch for:
- En passant discovered check
- Castling through/out of check
- Pawn promotion to all piece types
- Pin detection for en passant

#### MVP-3: Basic Search (Days 6–8)

##### Day 6: Evaluation

| File | Contents |
|------|----------|
| `traits.rs` | `Evaluator` trait definition (pure interface) |
| `material.rs` | Pure piece counting |
| `pst.rs` | Const piece-square tables |
| `simple.rs` | MVP evaluator (material + PST, composed) |

##### Day 7: Core Search

| File | Contents |
|------|----------|
| `negamax.rs` | Functional alpha-beta with `try_fold` |
| `ordering.rs` | MVV-LVA ordering (pure scoring) |

##### Day 8: Search Infrastructure

| File | Contents |
|------|----------|
| `quiescence.rs` | Capture-only search |
| `tt/table.rs` | Basic transposition table (interior mutability) |
| `context.rs` | `SearchContext` — explicit state, no globals |

#### MVP-4: UCI & Integration (Days 9–11)

##### Day 9: UCI Protocol

| File | Contents |
|------|----------|
| `parser.rs` | Pure command parsing |
| `handler.rs` | Command execution |
| `output.rs` | Pure info string formatting |

**Required UCI Commands:**

```
uci              → id, uciok
isready          → readyok
position [fen] [moves]
go depth N       → bestmove
go movetime N    → bestmove
go wtime/btime   → bestmove
stop             → bestmove
quit
```

##### Day 10: Engine Integration

| File | Contents |
|------|----------|
| `engine.rs` | Orchestrates components |
| `config.rs` | Immutable configuration |
| `builder.rs` | Functional builder |

##### Day 11: Testing

- Test with chess GUI (Arena, Cute Chess)
- Play games against Stockfish Level 1
- Basic time management

#### MVP-5: Stabilisation (Days 12–14)

| Day | Focus |
|-----|-------|
| 12 | Bug fixes, edge cases (stalemate, 50-move, repetition) |
| 13 | Performance profiling, obvious optimisations |
| 14 | Documentation, clean up, **TAG v0.1.0** |

#### MVP Completion Checklist

- [ ] Perft passes all test positions
- [ ] Can play complete games via UCI
- [ ] Handles all draw conditions
- [ ] Doesn't crash on any legal position
- [ ] Searches at least 500k nodes/second
- [ ] Beats Stockfish Level 1 consistently

---

### Phase 2: Search Enhancements (Days 15–21)

**Goal:** Dramatically improve search efficiency
**Target Strength:** ~1800–2000 ELO (+300–500)

#### Implementation Schedule

| Day | Feature | ELO Gain | Files |
|-----|---------|----------|-------|
| 15 | PVS + Killer Moves | +90 | `pvs.rs`, `killers.rs` |
| 16 | History Heuristic | +30 | `history.rs` |
| 17 | Null Move Pruning | +80 | `pruning/null_move.rs` |
| 18 | Late Move Reductions | +100 | `pruning/lmr.rs` |
| 19 | Futility + Aspiration | +60 | `pruning/futility.rs` |
| 20 | SEE + Check Extensions | +80 | `see.rs` |
| 21 | Razoring + Testing | +20 | `pruning/razoring.rs` |

#### New Files

```
search/
├── ordering/
│   ├── killers.rs      # NEW — interior mutability (Cell<Move>)
│   ├── history.rs      # NEW — interior mutability (AtomicI16)
│   └── see.rs          # NEW — pure
├── pruning/
│   ├── mod.rs
│   ├── null_move.rs    # NEW
│   ├── lmr.rs          # NEW
│   ├── futility.rs     # NEW
│   └── razoring.rs     # NEW
└── pvs.rs              # NEW
```

#### Phase 2 Completion Checklist

- [ ] Search reaches depth 12+ in reasonable time
- [ ] NPS improved to 1M+
- [ ] Beta cutoff on first move >85%
- [ ] Beats Stockfish Level 3

---

### Phase 3: Evaluation Depth (Days 22–28)

**Goal:** Understand positions better
**Target Strength:** ~2100–2200 ELO (+100–200)

#### Implementation Schedule

| Day | Feature | ELO Gain | Files |
|-----|---------|----------|-------|
| 22–23 | Pawn Structure | +40 | `pawns.rs` |
| 24–25 | King Safety | +50 | `king_safety.rs` |
| 26 | Mobility + Pieces | +55 | `mobility.rs`, `pieces.rs` |
| 27 | Tapered Evaluation | +30 | `tapered.rs` |
| 28 | Endgame Knowledge | +25 | `endgame.rs` |

#### Evaluation Components Detail

**Pawn Structure (pure, cached via pawn hash):**
- Doubled pawns: −15 per pawn
- Isolated pawns: −20 per pawn
- Backward pawns: −10 per pawn
- Passed pawns: +20 to +120 (rank-based)
- Connected passed: additional +15

**King Safety (pure):**
- Pawn shield: +5 per pawn in front of king
- Open file next to king: −25
- Enemy pieces attacking king zone: −10 per attack unit

**Mobility (pure, iterator-based):**
- Knight: +4 per square
- Bishop: +3 per square
- Rook: +2 per square
- Queen: +1 per square

```rust
pub fn mobility_score(pos: &Position) -> Score {
    let us = pos.side_to_move();
    let them = us.flip();

    let our_mobility: i32 = PieceType::ALL.iter()
        .flat_map(|&pt| pos.pieces(us, pt).iter_squares())
        .map(|sq| count_attacks(pos, sq))
        .sum();

    let their_mobility: i32 = PieceType::ALL.iter()
        .flat_map(|&pt| pos.pieces(them, pt).iter_squares())
        .map(|sq| count_attacks(pos, sq))
        .sum();

    Score((our_mobility - their_mobility) as i16)
}
```

#### Phase 3 Completion Checklist

- [ ] Evaluation trace shows all components
- [ ] Pawn structure cached (pawn hash table)
- [ ] King safety triggers correct behaviour
- [ ] Endgames played accurately

---

### Phase 4: Performance & Polish (Days 29–35)

**Goal:** Professional-grade implementation
**Target Strength:** ~2200–2300 ELO (+100)

#### Performance Optimisations

| Day | Optimisation | Impact |
|-----|-------------|--------|
| 29 | PGO, TT bucket strategy | +5–10% speed, +20 ELO |
| 30 | Prefetch, cache alignment | +10% speed |
| 31 | SIMD for eval | +10% speed |
| 32–33 | Lazy SMP | +100 ELO |

#### Quality Improvements

| Day | Feature |
|-----|---------|
| 34 | Pondering, MultiPV, UCI options |
| 35 | CI/CD, benchmark regression tests |

#### Lazy SMP Implementation

```
Main Thread:
├── Manages UCI communication
├── Starts/stops worker threads
└── Collects best move

Worker Threads:                              (functional benefit: immutable
├── Each searches same position               positions = no locks needed
├── Shared transposition table                for position sharing)
├── Slightly different search parameters
└── First to finish "wins"
```

#### Phase 4 Completion Checklist

- [ ] Multi-threaded search working
- [ ] NPS >2M single-threaded
- [ ] All UCI options implemented
- [ ] CI runs perft + bench on every commit

---

### Phase 5: Advanced Extensions (Days 36+)

**Goal:** Competitive engine
**Target Strength:** 2400+ ELO

#### Path A: NNUE Evaluation (+200–300 ELO)

```
eval/nnue/
├── mod.rs
├── network.rs        # Network architecture
├── accumulator.rs    # Incremental updates (SIMD)
├── simd.rs           # SIMD inference
└── weights.bin       # Trained weights

Time: 2-3 weeks | Difficulty: High
```

#### Path B: Syzygy Tablebases (+50 ELO)

```
tablebases/
├── mod.rs
├── probe.rs          # Tablebase probing
└── wdl.rs            # Win/Draw/Loss handling

Time: 3-5 days | Difficulty: Medium (use fathom library)
```

#### Path C: Opening Book (+30 ELO)

```
book/
├── mod.rs
├── polyglot.rs       # Standard format
└── learning.rs       # Learn from games

Time: 2-3 days | Difficulty: Easy
```

#### Path D: Self-Play Training (+100–200 ELO)

```
training/
├── mod.rs
├── selfplay.rs       # Game generation
├── spsa.rs           # Parameter tuning
└── texel.rs          # Eval weight tuning

Time: 1-2 weeks | Difficulty: Medium
```

#### Path E: MCTS/Hybrid Search

```
search/mcts/
├── mod.rs
├── tree.rs
├── policy.rs
└── hybrid.rs         # Combine with alphabeta

Experimental. Unknown ELO impact.
```

---

## 10. Technical Specifications

### 10.1 Data Structures

#### Move (`u16`)

```
Bits 0-5:   From square (0-63)
Bits 6-11:  To square (0-63)
Bits 12-15: Flags
```

#### MoveList

```rust
pub struct MoveList {
    moves: [Move; 256],  // Stack-allocated, no heap
    len: u8,
}
```

#### Position (136 bytes, `#[repr(C)]`)

```rust
pub struct Position {
    pieces: [Bitboard; 12],      // 96 bytes
    occupancy: [Bitboard; 3],    // 24 bytes
    hash: ZobristHash,           // 8 bytes
    side_to_move: Color,         // 1 byte
    castling: CastlingRights,    // 1 byte
    en_passant: Option<Square>,  // 2 bytes
    halfmove_clock: u8,          // 1 byte
    fullmove_number: u16,        // 2 bytes
    _padding: u8,                // 1 byte
}
```

#### TTEntry (16 bytes, cache-aligned)

```rust
#[repr(C, align(16))]
pub struct TTEntry {
    hash_verify: u32,  // 4 bytes
    score: i16,        // 2 bytes
    best_move: Move,   // 2 bytes
    depth: u8,         // 1 byte
    flag: TTFlag,      // 1 byte (Exact, LowerBound, UpperBound)
    age: u8,           // 1 byte
    _padding: [u8; 5], // 5 bytes
}
```

### 10.2 Constants

```rust
pub const PAWN_VALUE: Score = Score(100);
pub const KNIGHT_VALUE: Score = Score(320);
pub const BISHOP_VALUE: Score = Score(330);
pub const ROOK_VALUE: Score = Score(500);
pub const QUEEN_VALUE: Score = Score(900);

pub const MATE: Score = Score(30000);
pub const MATE_IN_MAX: Score = Score(30000 - 100);
pub const DRAW: Score = Score(0);
pub const INFINITY: Score = Score(32000);

pub const MAX_PLY: usize = 128;
pub const MAX_DEPTH: u8 = 100;
pub const DEFAULT_TT_SIZE_MB: usize = 64;
```

### 10.3 UCI Options

| Option | Type | Default | Range |
|--------|------|---------|-------|
| Hash | spin | 64 | 1–16384 |
| Threads | spin | 1 | 1–256 |
| Move Overhead | spin | 10 | 0–5000 |
| SyzygyPath | string | "" | — |
| Ponder | check | false | — |
| MultiPV | spin | 1 | 1–500 |

---

## 11. Testing Strategy

### 11.1 Perft Testing

**Purpose:** Validate move generation correctness

```rust
#[test]
fn perft_startpos() {
    let pos = Position::from_fen(STARTPOS);
    assert_eq!(perft(&pos, 1), 20);
    assert_eq!(perft(&pos, 2), 400);
    assert_eq!(perft(&pos, 3), 8_902);
    assert_eq!(perft(&pos, 4), 197_281);
    assert_eq!(perft(&pos, 5), 4_865_609);
}
```

**Standard Test Positions:**

| Name | FEN | Depth | Nodes |
|------|-----|-------|-------|
| Startpos | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | 6 | 119,060,324 |
| Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -` | 5 | 193,690,690 |
| Position 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -` | 7 | 178,633,661 |
| Position 4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | 6 | 706,045,033 |
| Position 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | 5 | 89,941,194 |

### 11.2 EPD Test Suites

**Purpose:** Validate tactical and positional understanding

- **WAC (Win at Chess):** 300 tactical positions
- **ECM (Encyclopedia of Chess Middlegames):** Complex middlegame positions
- **STS (Strategic Test Suite):** Positional understanding

```rust
#[test]
fn wac_suite() {
    let suite = load_epd("resources/epd/wac.epd");
    let mut correct = 0;

    for position in suite {
        let result = engine.search(position, depth: 10);
        if position.best_moves.contains(&result.best_move) {
            correct += 1;
        }
    }

    assert!(correct >= 250, "WAC score: {}/300", correct);
}
```

### 11.3 Benchmark Command

**Purpose:** Deterministic node count for regression testing

```rust
pub fn bench() -> u64 {
    let positions = BENCH_POSITIONS;
    let mut total_nodes = 0;

    for fen in positions {
        let pos = Position::from_fen(fen);
        let result = search(&pos, depth: 12);
        total_nodes += result.nodes;
    }

    total_nodes
}
```

### 11.4 Self-Play Testing

**Purpose:** ELO estimation and regression detection

```bash
cutechess-cli \
    -engine name=dev cmd=./target/release/chess-engine \
    -engine name=baseline cmd=./baseline/chess-engine \
    -each tc=10+0.1 \
    -rounds 1000 \
    -pgnout games.pgn \
    -recover
```

### 11.5 Mock-Based Unit Testing

Pure functions are trivially testable; use mocks only where controlled mutation exists:

```rust
struct MockEvaluator { scores: HashMap<u64, Score> }

impl Evaluator for MockEvaluator {
    fn evaluate(&self, pos: &Position) -> Score {
        self.scores.get(&pos.hash().0).copied().unwrap_or(Score::DRAW)
    }
}

#[test]
fn test_search_finds_mate() {
    let mock_eval = MockEvaluator::new();
    let engine = EngineBuilder::new().with_evaluator(mock_eval).build();
    let pos = Position::from_fen("mate-in-2-fen");
    let result = engine.search(&pos, depth: 4);
    assert!(result.score >= MATE_IN_MAX);
}
```

---

## 12. Performance Targets

### 12.1 Speed Metrics

| Metric | MVP | Phase 2 | Phase 4 | Target |
|--------|-----|---------|---------|--------|
| Perft NPS | 50M | 80M | 100M | 100M+ |
| Search NPS | 500K | 1M | 2M | 2M+ |
| Time to depth 10 | 5s | 2s | 1s | <1s |

### 12.2 Search Quality Metrics

| Metric | MVP | Phase 2 | Target |
|--------|-----|---------|--------|
| Beta cutoff 1st move | 70% | 85% | 90% |
| TT hit rate | 40% | 60% | 70% |
| Average branching factor | 6 | 3 | 2.5 |

### 12.3 Strength Progression

| Phase | Estimated ELO | Benchmark |
|-------|---------------|-----------|
| MVP | 1400–1600 | Beats beginner programs |
| Phase 2 | 1800–2000 | Beats club players |
| Phase 3 | 2100–2200 | Beats strong amateurs |
| Phase 4 | 2200–2300 | Competitive with mid-tier engines |
| Phase 5 | 2400+ | Competitive with strong engines |

---

## 13. Development Workflow

### 13.1 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test --all
      - run: cargo run --release -- bench
```

---

## 14. Risk Management

### 14.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Move generation bugs | High | Critical | Extensive perft testing |
| Search bugs | High | High | Compare with Stockfish |
| Performance regression | Medium | Medium | CI benchmarks |
| Magic bitboard complexity | Medium | Medium | Use pre-computed magics |

### 14.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Debugging takes longer | High | Medium | Buffer time in schedule |
| Scope creep | Medium | Medium | Strict MVP definition |
| Burnout | Medium | High | Sustainable pace |

### 14.3 Common Debugging Pitfalls

**Move Generation:**
- En passant discovered check
- Castling through/out of check
- Promotion to all piece types
- Pin detection edge cases

**Search:**
- Transposition table type-1 errors
- Score bounds confusion (fail-soft vs fail-hard)
- Mate score adjustment across plies
- Repetition detection bugs

### 14.4 Recovery Strategies

- **Perft fails:** Stop all other work, debug until fixed
- **Search returns bad moves:** Add tracing, compare with known-good engine
- **Performance regression:** Git bisect to find culprit
- **Mysterious crashes:** Run with ASAN/MSAN

---

## 15. Quick Reference

### 15.1 What to Build When

| If you want to... | Build this first |
|-------------------|------------------|
| Play a game | MVP (Phase 1) |
| Beat weak engines | Search enhancements (Phase 2) |
| Play positionally well | Evaluation (Phase 3) |
| Use multiple cores | SMP (Phase 4) |
| Reach 2400+ ELO | NNUE (Phase 5) |
| Perfect endgames | Syzygy (Phase 5) |

### 15.2 File Quick Reference

| Need to modify... | Look in... |
|-------------------|------------|
| Piece values | `crates/eval/src/classical/material.rs` |
| Move ordering | `crates/search/src/ordering/` |
| Pruning parameters | `crates/search/src/pruning/` |
| UCI commands | `crates/uci/src/handler.rs` |
| Time management | `crates/time/src/` |
| Search context | `crates/search/src/context.rs` |

### 15.3 Useful Commands

```bash
# Run tests
cargo test --all

# Run benchmarks
cargo bench

# Build release
cargo build --release

# Run engine
./target/release/chess-engine

# Profile (Linux)
perf record ./target/release/chess-engine bench
perf report

# Check for undefined behavior
RUSTFLAGS="-Z sanitizer=address" cargo +nightly run

# Run perft
echo "position startpos\ngo perft 5" | ./target/release/chess-engine
```

### 15.4 External Resources

- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [Stockfish Source](https://github.com/official-stockfish/Stockfish)
- [Cute Chess GUI](https://cutechess.com/)
- [CCRL Rating Lists](https://www.computerchess.org.uk/ccrl/)
- [Syzygy Tablebases](https://syzygy-tables.info/)

---

## 16. Appendices

### Appendix A: Timeline Visualisation

```
┌──────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION TIMELINE                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: MVP                        Days 1-14   (~1500 ELO)         │
│  ████████████████████████████████                                    │
│  │ Types │ Board │ MoveGen │ Search │ UCI │ Stabilise │              │
│                                                                      │
│  PHASE 2: Search Enhancements        Days 15-21  (~2000 ELO)         │
│                                  ██████████████                      │
│                                  │ PVS │ NMP │ LMR │ SEE │           │
│                                                                      │
│  PHASE 3: Evaluation                 Days 22-28  (~2200 ELO)         │
│                                                  ██████████████      │
│                                                  │ Pawns │ KS │ Mob ││
│                                                                      │
│  PHASE 4: Performance                Days 29-35  (~2300 ELO)         │
│                                                              █████████│
│                                                              │ SMP │ │
│                                                                      │
│  PHASE 5: Extensions                 Days 36+    (2400+ ELO)         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─►            │
│  │ NNUE │ Syzygy │ Books │ Training │ ...                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Appendix B: Success Metrics

| Milestone | Criteria | Tag |
|-----------|----------|-----|
| **MVP Complete** | Passes perft, plays via UCI, beats SF Level 1 | v0.1.0 |
| **Search Complete** | 1M+ NPS, >85% first-move cutoff, beats SF Level 3 | v0.2.0 |
| **Eval Complete** | All eval components traced, endgames clean | v0.3.0 |
| **Production Ready** | Multi-threaded, CI green, documented | v1.0.0 |
| **Competitive** | 2400+ ELO on CCRL or similar | v2.0.0 |

### Appendix C: Perft Debugging Checklist

When perft fails, check these in order:

- [ ] **Pawn moves:** single push, double push (only from rank 2/7), captures (diagonal only), en passant (including discovered check edge case), promotions (all 4 piece types), promotion captures
- [ ] **Castling:** king-side rights, queen-side rights, path clear, not through check, not out of check, rook present
- [ ] **King moves:** not moving into check, all 8 directions
- [ ] **Sliding pieces:** blocking by own pieces, blocking by enemy pieces, captures
- [ ] **Pins:** absolute pins (can only move along pin ray), en passant pin edge case
- [ ] **Check evasions:** king moves away, blocking, capturing checker, double check (only king can move)

### Appendix D: Sample Piece-Square Tables

#### Pawn (Middlegame)

```
  0,  0,  0,  0,  0,  0,  0,  0,
 50, 50, 50, 50, 50, 50, 50, 50,
 10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10, 25, 25, 10,  5,  5,
  0,  0,  0, 20, 20,  0,  0,  0,
  5, -5,-10,  0,  0,-10, -5,  5,
  5, 10, 10,-20,-20, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0,
```