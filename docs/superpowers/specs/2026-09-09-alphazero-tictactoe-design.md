# AlphaZero-Style Tic-Tac-Toe Player — Design

## Purpose

A from-scratch, dependency-free C++ implementation of the AlphaZero
algorithm (self-play + PUCT/MCTS guided by a neural network) applied
to tic-tac-toe, built as a learning project. The goal is to understand
every moving part of the training loop by implementing it directly,
rather than to produce a general-purpose game engine.

## Success Criteria

- Training converges to optimal play: after enough self-play
  iterations, the trained agent draws (never loses) against a perfect
  minimax player, playing both as X and as O.
- All components (board, network, MCTS, self-play, training,
  evaluation, CLI) are implemented and testable independently.
- No external dependencies beyond the C++ standard library.

## Non-Goals

- Generalizing beyond tic-tac-toe (no plans to support other games).
- Implementing the AlphaZero "arena" network-promotion step (see
  Rationale below) — may be added later as a follow-up exploration,
  not part of this spec.
- GPU acceleration, batched/parallel MCTS, or performance tuning
  beyond what's needed to train in reasonable time on a single
  machine — tic-tac-toe's state space is tiny (5,478 legal states),
  so this is not a concern.
- MuZero-style learned dynamics — the game rules (legal moves,
  transitions, terminal detection) are given directly to MCTS, not
  learned, matching AlphaZero (not MuZero).

## Architecture

A shared library (`libaz`) with independent, single-purpose
components, consumed by three executables (`train`, `evaluate`,
`play_cli`).

```
train ----\
evaluate --+--> libaz (board, network, mcts, replay_buffer, selfplay, minimax)
play_cli -/
```

### Components

**`board`**
3x3 grid state (9 cells, each empty/X/O), current player to move.
Responsibilities: apply a move, list legal moves, detect
win/draw/ongoing. Pure value type — no I/O, no dependencies on other
components.

**`network`**
A small MLP, hand-written (forward pass and backprop implemented
directly — no autograd library).
- Input: 18 floats — two 9-element planes encoding the board from the
  perspective of the player to move ("my stones", "opponent stones").
  This canonical (player-relative) encoding means the network only
  ever has to reason about "the player to move," not X vs O, which
  halves what it has to learn.
- One hidden layer (64 units, ReLU) — sized generously for a state
  space this small; can shrink if training is too slow.
- Two output heads:
  - Policy: 9 logits → softmax, illegal moves masked out and
    renormalized.
  - Value: 1 scalar → tanh, in [-1, 1], from the perspective of the
    player to move.
- Loss: mean-squared error on value + cross-entropy on policy
  (matching AlphaZero's combined loss), no separate regularization
  term needed at this scale.
- Save/load weights to/from a flat file (simple binary format) so
  `train`, `evaluate`, and `play_cli` can share checkpoints.

**`mcts`**
PUCT search per AlphaZero: no random rollouts — leaf values come
directly from the network's value head, leaf priors from the policy
head. Runs a configurable number of simulations per move and returns
a visit-count distribution over legal moves, used both to select the
actual move (via temperature-based sampling, see `selfplay`) and as
the policy training target.

**`replay_buffer`**
Fixed-capacity store of `(board encoding, MCTS policy, game outcome)`
training examples, with random-batch sampling. Simple ring buffer —
no prioritization.

**`selfplay`**
Plays complete games using MCTS+network only (no external opponent).
Move selection uses temperature-based sampling from the MCTS visit
distribution during the opening moves (for training-data diversity)
and greedy (argmax) selection afterward — matching AlphaZero's
approach. Each finished game's positions are pushed into the replay
buffer, with the value target set to the actual game outcome from
each position's player-to-move perspective.

**`minimax`**
Exhaustive perfect-play solver. Tic-tac-toe's full game tree is small
enough to search exhaustively with no pruning. Used only for
evaluation (measuring convergence), never for training or as a
training opponent.

### Executables

**`train`**
Orchestrates the loop: run self-play games → collect examples into
the replay buffer → sample batches and train the network → repeat for
a configured number of iterations. Periodically (every K iterations):
saves a checkpoint to disk, and plays a small evaluation match against
`minimax`, printing win/draw/loss counts so convergence is visible
during training.

**`evaluate`**
Loads a checkpoint, plays a configurable number of games against
`minimax` (split between playing as X and as O), reports aggregate
results.

**`play_cli`**
Loads a checkpoint, runs an interactive terminal game where a human
plays against the trained agent. Rejects illegal input and re-prompts
rather than crashing.

## Data Flow

1. `train` calls `selfplay` repeatedly, each self-play game using the
   current network to guide `mcts`.
2. Finished games' positions land in `replay_buffer`.
3. `train` samples batches from `replay_buffer` and runs gradient
   steps on `network`.
4. Every K iterations, `train` saves a checkpoint and runs a short
   `evaluate`-equivalent pass against `minimax`, logging results.
5. After training, `evaluate` and `play_cli` independently load a
   saved checkpoint — they do not depend on `train` being run in the
   same process.

## Error Handling

The domain is small and deterministic, so error handling stays
minimal:
- `board`: applying an illegal move is a programming error (asserted,
  not handled gracefully) inside `mcts`/`selfplay`, since those only
  ever generate legal moves internally.
- `play_cli`: user-typed moves are the one real "untrusted input"
  boundary — illegal or malformed input is rejected with a message
  and re-prompted, never asserted.
- Checkpoint load failure (missing/corrupt file) in `evaluate` /
  `play_cli`: fail fast with a clear error message; no fallback to
  random weights.

## Testing

Plain `assert`-based test executables (no framework, consistent with
the zero-dependency approach):
- `board`: win/draw detection across known positions, legal move
  generation.
- `network`: forward-pass output shapes and ranges (policy sums to 1
  over legal moves, value in [-1, 1]); a small gradient-check style
  test (numerical vs. analytical gradient) to catch backprop bugs.
- `mcts`: on a known forced-win/forced-block position, confirm the
  visit distribution concentrates on the correct move (using a
  hand-constructed "oracle" network for this test rather than a
  trained one, so the test doesn't depend on training having
  succeeded).
- Integration: a short `train` run (few iterations, small simulation
  count) completes without crashing and produces a checkpoint that
  `evaluate` and `play_cli` can load.

## Build

CMake, C++17, no external dependencies. One `libaz` static library
target; `train`, `evaluate`, `play_cli`, and the test executables all
link against it.

## Rationale: Skipping the Arena/Promotion Step

Full AlphaZero only replaces the "best" network with a newly trained
one if it wins a gated match against it, to guard against training
instability on large, hard domains. Tic-tac-toe's state space is small
enough that continuous training (no gating) reliably converges, and
periodic evaluation against a perfect `minimax` player gives a direct,
unambiguous convergence signal (unlike chess/Go, where no perfect
oracle is available and self-comparison is the only option). Adding
gating later remains a reasonable follow-up exploration but isn't
needed to meet this project's success criteria.
