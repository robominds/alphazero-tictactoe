# AlphaZero-Style Tic-Tac-Toe

A from-scratch, dependency-free C++17 implementation of the AlphaZero
algorithm — self-play guided by PUCT/MCTS and a hand-written neural
network (forward pass and backpropagation written by hand, no autograd
library) — applied to tic-tac-toe as a learning project.

See [`docs/superpowers/specs/2026-09-09-alphazero-tictactoe-design.md`](docs/superpowers/specs/2026-09-09-alphazero-tictactoe-design.md)
for the design rationale and [`docs/superpowers/plans/2026-09-09-alphazero-tictactoe.md`](docs/superpowers/plans/2026-09-09-alphazero-tictactoe.md)
for the implementation plan.

For a guided walkthrough of the algorithm itself — board encoding, the
network, MCTS/PUCT, self-play, and training, with real code and RL
fundamentals explained along the way — see
[`docs/algorithm-explained.md`](docs/algorithm-explained.md) (plain
markdown) or open [`docs/algorithm-explained.html`](docs/algorithm-explained.html)
in a browser for the illustrated version with diagrams.

## Build

Requires CMake 3.16+ and a C++17 compiler. No external dependencies.

```sh
mkdir build && cd build
cmake ..
cmake --build .
ctest --output-on-failure   # run the test suite (7 tests)
```

## Usage

Three executables are produced in `build/`:

```sh
# Train from scratch. Periodically prints self-play/training progress and
# evaluates against a perfect minimax player; saves checkpoints along the way.
./train [iterations] [checkpoint-path]
# defaults: 200 iterations, checkpoint.bin

# Score a saved checkpoint against perfect minimax play (as both X and O).
./evaluate <checkpoint-path> [games-per-side]
# default: 50 games per side

# Play interactively against a saved checkpoint (you are X).
./play_cli <checkpoint-path>
```

Typical session:

```sh
./train 300 checkpoint.bin
./evaluate checkpoint.bin
./play_cli checkpoint.bin
```

### What "trained" looks like

Training converges to optimal (drawing) play against minimax — with root
Dirichlet noise during self-play (see below), a real run reached
`draws=40 losses=0` (all 40 games, both sides) after just **20 iterations**.

### Debugging a checkpoint

`MCTS::run` guides self-play exploration with Dirichlet noise mixed into
the root's priors (`SelfPlayConfig::dirichletAlpha`/`dirichletEpsilon`, on
by default), so search can't permanently starve a move the network is
(possibly wrongly) confident is bad. Without it, a network can get stuck
with a genuine blind spot: confidently losing every game from one side
because search never visits the one move that mattered enough to correct
it. `tools/diag_eval.cpp` (built as `./diag_eval <checkpoint>`) plays one
game as X and one as O against minimax, printing every move and MCTS
visit distribution — useful for tracing exactly where and why a
checkpoint loses.

## Project layout

```
include/az/   public headers for each component
src/          implementations (board, minimax, network, mcts, replay
              buffer, self-play, minimax-eval helper)
apps/         the three executables (train, evaluate, play_cli)
tools/        diag_eval, a per-side game-transcript diagnostic
tests/        assert-based test executables (one per component) plus
              tests/integration_smoke.sh, an end-to-end pipeline check
```

Deliberately out of scope (see the spec's Non-Goals): AlphaZero's
arena/network-promotion step, and MuZero-style learned game dynamics —
tic-tac-toe's state space is small enough that continuous training plus
periodic minimax evaluation gives a clear, unambiguous convergence signal
without either.

## Authorship

Mark Castelluccio, designed and implemented with [Claude Code](https://claude.com/claude-code).
