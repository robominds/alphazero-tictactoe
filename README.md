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
ctest --output-on-failure   # run the test suite (8 tests)
```

## Usage

Three executables are produced in `build/`:

```sh
# Train from scratch. Periodically prints self-play/training progress and
# evaluates against a perfect minimax player; saves checkpoints along the way.
./train [iterations] [checkpoint-path]
# defaults: 200 iterations, checkpoint.bin

# Score a saved checkpoint against perfect minimax play (as both X and O).
# Minimax picks at random among equally good moves, so games vary.
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

Against perfect play the best a network can do is draw every game, so the
number to watch is the draw rate. Read it against a baseline: search alone
is strong in a game this small, and an **untrained** network with
100-simulation search already draws about 60–70% of games against
minimax. Improvement shows up as a draw rate above that.

With the default settings, training is not yet reliable. Four
200-iteration runs finished at 57, 63, 68 and 92 draws out of 100
(`./evaluate`, 50 games per side); most are no better than the untrained
baseline. A run with 10× the training steps per iteration and a 5× learning
rate (`trainStepsPerIteration` and `learningRate` in `apps/train.cpp`)
reached 91. No checkpoint tested so far is unbeatable: an opponent
that tries every reply can still beat each of them as O.

Earlier versions of this README reported `draws=40 losses=0` after 20
iterations. That figure came from an evaluation where minimax always broke
ties the same way. Greedy search is deterministic too, so every game on a
side was the same game, and the 40 games were really 2.

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
checkpoint loses. Minimax's tie-breaking is random, so rerun it to see
other lines.

### Keeping the explainer's citations honest

`docs/algorithm-explained.md` captions each code excerpt with the file and
line it came from, and those line numbers rot silently: any commit that
inserts a line above a quoted function invalidates a citation without
making the document look wrong. `tools/check_citations.py` verifies every
caption against the current source, and `--fix` rewrites the stale ones.
It needs only the standard library and is not part of the build.

```sh
python3 tools/check_citations.py        # report
python3 tools/check_citations.py --fix  # rewrite in place
```

## Project layout

```
include/az/   public headers for each component
src/          implementations (board, minimax, network, mcts, replay
              buffer, self-play, minimax-eval helper)
apps/         the three executables (train, evaluate, play_cli)
tools/        diag_eval, a per-side game-transcript diagnostic;
              check_citations.py, which verifies the explainer's
              file:line citations still point where they claim
tests/        assert-based test executables (one per component) plus
              tests/integration_smoke.sh, an end-to-end pipeline check
```

Deliberately out of scope (see the spec's Non-Goals): AlphaZero's
arena/network-promotion step, and MuZero-style learned game dynamics —
tic-tac-toe's state space is small enough that continuous training plus
periodic minimax evaluation gives a direct measure of progress without
either.

## Authorship

Mark Castelluccio, designed and implemented with [Claude Code](https://claude.com/claude-code).
