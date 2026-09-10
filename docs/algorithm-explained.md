# AlphaZero from Scratch

How a 9-square board, a 64-unit neural net, and a tree search taught
themselves to never lose — walked through against the actual C++ in this
repository, section by section, in the order the code actually runs.

*(A richer, illustrated version of this document — with diagrams and syntax
highlighting — is available as a published page; see the project README.)*

**Language:** C++17, zero dependencies · **Network:** 18 → 64 → {9, 1} ·
**Result:** `draws=40 / losses=0` vs. perfect play, sustained

## Contents

- [00 — The state](#00--the-state)
- [01 — The two-headed guess](#01--the-two-headed-guess)
- [02 — Search corrects the guess](#02--search-corrects-the-guess)
- [03 — Self-play & temperature](#03--self-play--temperature)
- [04 — Assigning credit](#04--assigning-credit)
- [05 — Learning from itself](#05--learning-from-itself)
- [06 — The full loop](#06--the-full-loop)
- [07 — Concept map](#07--concept-map)

---

## 00 — The state

**Concept: Markov decision process**

Every reinforcement learning problem starts with the same three
ingredients: a **state** the agent observes, a set of **actions** it can
take, and a **reward** it eventually receives. Tic-tac-toe is a two-player,
zero-sum, perfect-information Markov decision process small enough to see
whole — there are exactly 5,478 legal board positions — which is precisely
why it's a good place to learn the machinery before pointing it at
something you can't fully enumerate.

In this codebase the state is `az::Board`: nine cells, each
`Empty`/`X`/`O`, plus whose turn it is. The interesting design decision
isn't the grid — it's how that grid gets turned into numbers a neural
network can read.

```cpp
// 18 floats: [my stones (9), opponent stones (9)], from the
// perspective of playerToMove().
std::array<float, 18> Board::encode() const {
    std::array<float, 18> out{};
    Cell mine = toMove_;
    Cell theirs = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    for (int i = 0; i < 9; ++i) {
        out[i]     = (cells_[i] == mine)   ? 1.0f : 0.0f;
        out[9 + i] = (cells_[i] == theirs) ? 1.0f : 0.0f;
    }
    return out;
}
```
*src/board.cpp:62 — `Board::encode`*

Notice what's *not* in that encoding: the literal symbols X and O. The
network only ever sees "my stones" and "the opponent's stones," never
which physical mark they correspond to. This is a small instance of a
general RL technique — **state canonicalization** — collapsing states that
are strategically identical into one representation. Here it halves what
the network has to learn: a position is exactly as good for "me" whether
I'm playing X or O, so there's no reason to make the network discover that
symmetry from data when the encoding can hand it over for free.

**Example.** Position: X at square 0, O at square 4, O to move.

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| my stones (0–8), O's view    | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 |
| opp. stones (9–17), O's view | **1** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The same physical square (index 0, occupied by X) lands in a different
half of the 18-float vector depending on who's asking. If it were X's move
instead, this exact board would encode with the 1s swapped — X's stone
would appear in "my stones," O's in "opponent's."

> **Why this matters beyond tic-tac-toe.** Real AlphaZero does the same
> trick for chess and Go — the network always sees the board "from the
> side to move," and even flips the board orientation so it never has to
> separately learn White's and Black's strategy. Canonical state
> representations are one of the cheapest wins in applied RL: anything the
> encoding can guarantee, the network doesn't have to spend capacity
> discovering.

---

## 01 — The two-headed guess

**Concept: policy π(a\|s) and value V(s)**

Given a state, two questions matter: **which move looks good** (the
*policy*), and **how good is this position overall** (the *value*).
AlphaZero's central idea is to approximate both with one small function —
here, a hand-written multilayer perceptron with no automatic-differentiation
library behind it, because the whole point of this project was to write
the forward and backward passes by hand.

```
input (18)                     hidden (64, ReLU)                    heads
[my stones (9)]   ─┐                                            ┌─▶ policy (9)  → softmax → π(a|s)
[opp. stones (9)]  ─┴──▶  64 fully-connected units, ReLU  ───────┤     "which move looks good"
                                                                  └─▶ value (1)   → tanh    → V(s) ∈ [-1,1]
                                                                        "how good is this?"
```

One shared trunk, two heads. Both heads read the same 64-unit hidden
layer — the network learns one internal representation of "what's going on
in this position" and spends it two ways.

```cpp
Prediction Network::predict(const std::array<float, 18>& encodedBoard) const {
    // input(18) → hidden(64, ReLU)
    std::array<float, 64> a1{};
    for (int h = 0; h < 64; ++h) {
        float z = b1_[h];
        for (int i = 0; i < 18; ++i) z += w1_[h][i] * encodedBoard[i];
        a1[h] = z > 0.0f ? z : 0.0f;              // ReLU
    }
    // hidden(64) → policy logits(9) → softmax
    std::array<float, 9> logits{};
    for (int k = 0; k < 9; ++k) {
        float z = bPolicy_[k];
        for (int h = 0; h < 64; ++h) z += wPolicy_[k][h] * a1[h];
        logits[k] = z;
    }
    /* ... softmax(logits) → policy ... */

    // hidden(64) → value preactivation(1) → tanh
    float valuePre = bValue_;
    for (int h = 0; h < 64; ++h) valuePre += wValue_[h] * a1[h];
    float value = std::tanh(valuePre);

    return Prediction{policy, value};
}
```
*src/network.cpp:23 — `Network::predict` (softmax elided)*

Two details worth noticing, both standard RL/deep-learning choices rather
than accidents:

- **The value head ends in `tanh`, not a raw number.** Game outcomes here
  are win / draw / loss — `+1`, `0`, `-1` — so squashing the output into
  that exact range means the network's guess and the label it's trained
  against always live in the same units.
- **The policy head is a full softmax over all 9 cells**, even the ones
  that are already occupied. `predict()` has no idea which moves are
  legal — it's a pure function of the encoded board. Something downstream
  has to mask out illegal moves before trusting the distribution. That
  "something" is section 02.

> **Why a two-headed net at all.** Classic RL splits these into separate
> algorithms — a policy-gradient method learns π, a value-based method
> like Q-learning learns V or Q. AlphaZero's insight is that both
> quantities are useful for the *same* search (the policy tells search
> where to look first, the value tells it when to stop looking and just
> estimate), so training one shared network to predict both, with one
> shared representation, is both cheaper and — empirically — better than
> training two networks separately.

---

## 02 — Search corrects the guess

**Concept: planning, exploration vs. exploitation, bootstrapping**

A freshly initialized network's guesses are close to random. If the agent
just played whatever move the policy head liked best, it would play badly
forever — there's no learning signal telling the network it's wrong until
*something* more reliable checks its guesses against the actual rules of
the game. That something is **Monte Carlo Tree Search** (MCTS),
specifically the PUCT variant AlphaZero uses.

The idea: from the current position, run many short simulated lookaheads.
Each one picks a move, walks one ply deeper, and either hits a position
the network has already scored (use that estimate) or hits the actual end
of the game (use the real outcome — no estimate needed, because at a
terminal position the truth is known exactly). Every simulation's result
gets backed up the tree, refining a running estimate of "how good is each
of my *legal* moves, really" — an estimate strictly better than the
network's raw, unchecked guess.

Which move to explore next during search is the classic
**exploration/exploitation** trade-off, resolved here by the PUCT formula:

```
score(a) = Q(s,a) + c_puct · P(a|s) · sqrt(Σ_b N(s,b)) / (1 + N(s,a))
```

```cpp
float q = node.visitCounts[m] > 0 ? node.totalValue[m] / node.visitCounts[m] : 0.0f;
float u = cPuct_ * node.priors[m] * std::sqrt(totalVisits + 1e-8f) / (1 + node.visitCounts[m]);
float score = q + u;
```
*src/mcts.cpp:33 — `MCTS::selectChild`*

Read left to right: `Q(s,a)` is **exploitation** — the mean value this
move has actually backed up so far, i.e. "how well has this move
performed in simulations we've already run." The second term is
**exploration** — it starts large (when `N(s,a)`, the visit count, is 0)
and shrinks every time the move gets visited again, while staying larger
for moves the network's prior `P(a|s)` already likes. In plain terms:
*try what the network suggests first, but don't ignore a move forever
just because you haven't tried it yet.* `c_puct = 1.5` in this codebase
controls how strongly that exploration term counts against the
accumulated evidence in `Q`.

**Worked example** — the exact position from `tests/test_mcts.cpp`: X has
two in a row and playing square 2 wins outright.

```
                    root — X to move
                    X X · / O O · / · · ·
                    legal: 2, 5, 6, 7, 8
                    Q(root, move 2) → 1.00
                   /                                  \
          move 2                                        move 5  (one alternative)
   X X X / O O · / · · ·                         X X · / O O X / · · ·
   terminal — X wins                              not terminal — expand
   O to move — O has lost                          O to move — game continues
   leaf value = -1                                  leaf value ≈ v(network)

   back up: -(-1) = +1                              an untrained guess, not a
   one simulation is enough — the                    fact — and it will keep
   outcome is exact, not estimated                   changing as this subtree
                                                       gets revisited
```

Because the winning child is terminal, its true value is known
immediately — no network guess needed — and after only a handful of
simulations its backed-up `Q` at the root dwarfs every other move. Search
found the tactic; the untrained network's priors never could have.

The sign flip in "back up: `-(-1) = +1`" is the single most important
piece of bookkeeping in the whole search, and it's worth being explicit
about why it's there. Every value in this codebase is stored **from the
perspective of whoever is about to move** at that node — exactly like the
board encoding in section 00. But the player to move alternates every
ply. So a value that means "great for me" one level down means "terrible
for me" one level up, and the code negates it exactly once per ply on the
way back to the root:

```cpp
float MCTS::simulate(Node& node) {
    if (node.board.isTerminal()) {
        // the player to move at a terminal node never just won —
        // the win happened on the *previous* ply
        return node.board.outcome() == Outcome::Draw ? 0.0f : -1.0f;
    }
    if (!node.expanded) return expand(node);       // leaf: trust the network

    int move = selectChild(node);
    /* ... descend into node.children[move] ... */
    float childValue = simulate(*node.children[move]);
    float value = -childValue;                     // ← the flip
    node.visitCounts[move] += 1;
    node.totalValue[move]  += value;
    return value;
}
```
*src/mcts.cpp:50 — `MCTS::simulate`*

This is the same convention used independently in three places in this
codebase — search, the exhaustive minimax opponent used for evaluation,
and the value labels assigned during self-play (section 04) — and it has
to agree across all three or training silently learns the wrong thing.
It's exactly the kind of one-line bug that would still compile, still
run, and just quietly teach the network to prefer losing.

Once simulation stops, the fraction of total visits each legal move
received becomes both the move actually played *and* — separately — a
training target for the policy head. That second use is the deep idea
behind AlphaZero: **search is a policy-improvement operator**. Feed it a
mediocre policy (the raw network), get back a better one (the visit
distribution), for free, just by spending more compute at decision time.
Training then tries to make the cheap network alone predict what the
expensive search found — so over many iterations, the network gets good
enough that search needs to correct it less.

---

## 03 — Self-play & temperature

**Concept: trajectory generation, exploration via sampling**

Supervised learning needs labeled examples. AlphaZero's trick is that it
never needs a human to provide them — it generates its own by playing
itself, using exactly the network-plus-search combination from sections
01–02 for *both* sides of the board:

```cpp
while (!board.isTerminal()) {
    MCTS mcts(network, config.numSimulations, config.cPuct);
    float temperature = (ply < config.temperatureMoves) ? 1.0f : 0.0f;
    MCTSResult result = mcts.run(board, temperature);
    pending.push_back({board.encode(), result.visitDistribution, board.playerToMove()});
    board = board.applyMove(result.selectedMove);
    ++ply;
}
```
*src/selfplay.cpp:17 — `playSelfPlayGame`*

The `temperature` parameter controls a second exploration/exploitation
trade-off, this time over which move actually gets *played* in the
generated game (not which move search merely considers):

- **temperature = 1.0** (the first `temperatureMoves = 2` plies): sample a
  move with probability proportional to its raw visit count. A move
  visited twice as often is twice as likely to be picked — but every
  legal move with at least one visit has *some* chance. This is what
  makes self-play generate varied openings instead of replaying the
  identical game forever.
- **temperature = 0** (every ply after that): always play the single
  most-visited move, deterministically. Once the game is a few moves in,
  there's no more benefit to exploring — search has already done its
  job, so play its best answer.

> **The RL fundamental hiding in one config struct.**
> `SelfPlayConfig::temperatureMoves` is a two-line answer to a question
> every RL algorithm has to answer somehow: *how do you keep generating
> diverse experience without wrecking the quality of the data you learn
> from?* Too little exploration and the agent only ever practices one
> line of play, never discovering better ones. Too much and most of its
> games are close to random, diluting the training signal with noise.
> Annealing temperature from "explore" to "exploit" partway through each
> trajectory is a small, concrete instance of that trade-off, tuned for a
> game that's usually over in five to nine moves.

---

## 04 — Assigning credit

**Concept: Monte Carlo return, credit assignment**

A self-play game produces a sequence of positions, but a single scalar
outcome — someone won, someone lost, or it was a draw — only arrives at
the very end. **Credit assignment** is the general RL problem of turning
that one end-of-episode signal into a training label for every state that
led up to it. Tic-tac-toe's answer is about as direct as credit
assignment ever gets:

```cpp
Outcome outcome = board.outcome();               // known only once the game ends
for (const auto& p : pending) {
    float z;
    if (outcome == Outcome::Draw) {
        z = 0.0f;
    } else {
        bool playerWon = (outcome == Outcome::XWins && p.playerToMove == Cell::X) ||
                          (outcome == Outcome::OWins && p.playerToMove == Cell::O);
        z = playerWon ? 1.0f : -1.0f;
    }
    buffer.add(TrainingExample{p.encoded, p.policy, z});
}
```
*src/selfplay.cpp:28 — `playSelfPlayGame`*

Every position recorded earlier in the game is revisited after it's over
and stamped with `z` — the real, final result, seen from *that position's
own player-to-move at the time* (not the last player to move in the
finished game; see the sign-flip discussion in section 02, the same
perspective convention applies here). This is a **Monte Carlo return**:
an unbiased sample of a state's true value, computed by actually playing
to the end rather than estimating. It's high-variance from any single
game — a position that was objectively fine might get labeled `−1` just
because of what happened many moves later — but averaged over the
thousands of games a training run produces, it converges to the truth.

| Target | Where it comes from | What kind of estimate |
|---|---|---|
| `targetPolicy` | MCTS visit distribution (section 02) | Bootstrapped — built from other estimates, refined in real time during search |
| `targetValue` (`z`) | The finished game's actual outcome (this section) | Monte Carlo — the ground truth for this one rollout, no estimation involved |

Mixing a bootstrapped policy target with a Monte-Carlo value target in
the same training example is itself a well-known RL design choice — it's
what AlphaZero (and its predecessor, TD-learning methods like TD-Gammon)
do to get faster-converging value estimates than pure Monte Carlo alone,
without the network ever having to *be* the search.

---

## 05 — Learning from itself

**Concept: supervised learning on self-generated labels**

Once a replay buffer of `(board, MCTS policy, z)` triples exists, training
is no longer really "reinforcement learning" in the moment-to-moment
sense — it's ordinary supervised learning, minimizing a loss against
labels the system manufactured for itself two sections ago:

```
L = (v − z)² + −Σ_a π_MCTS(a) · log p_network(a)
```

The first term is ordinary mean-squared error, pulling the value head's
guess `v` toward the Monte Carlo outcome `z`. The second is cross-entropy,
pulling the policy head's distribution toward the (better-informed)
search-derived distribution. Because the network's forward pass in this
project is hand-written, so is its backward pass — every gradient below
is the chain rule, spelled out, with no autograd library computing it for
you:

```cpp
float dValuePre = 2.0f * (value - example.targetValue) * (1.0f - value * value);   // d/dv (v-z)², through tanh'
for (int k = 0; k < 9; ++k) dLogits[k] = policy[k] - example.targetPolicy[k];       // softmax+cross-entropy gradient

// both heads feed back into the one shared hidden layer:
for (int h = 0; h < 64; ++h) {
    float grad = dValuePre * wValue_[h];
    for (int k = 0; k < 9; ++k) grad += dLogits[k] * wPolicy_[k][h];
    dA1[h] = grad;
}
```
*src/network.cpp:99 — `Network::trainStep` (weight-gradient accumulation elided)*

That accumulation step is where the "two heads, one trunk" architecture
from section 01 earns its keep: the hidden layer's gradient is the *sum*
of what both heads want it to change, so a single backward pass improves
the shared representation for both jobs at once. Gradients are averaged
over a batch of `batchSize = 32` examples and applied with plain
stochastic gradient descent at `learningRate = 0.01` — no momentum, no
Adam, nothing beyond the update rule `w −= learningRate · gradient`. It
doesn't need more than that at this scale.

---

## 06 — The full loop

**Concept: the training loop, evaluation without self-play bias**

```
   ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
   │  self-play   │ ───▶ │  replay buffer  │ ───▶ │    train     │
   │ 25 games/it. │      │ capacity 10,000 │      │ 20×batch 32  │
   └──────────────┘      └─────────────────┘      └──────┬───────┘
          ▲                                               │
          └──────────── updated network plays next ───────┘
                                                            │
                                                  (every 10 iterations)
                                                            ▼
                                                    ┌────────────────┐
                                                    │    evaluate    │
                                                    │  vs. minimax   │
                                                    └────────────────┘
```

Everything on the self-play → buffer → train loop trains the network
against itself. The evaluate branch never does — `evaluateAgainstMinimax`
is the one place the network is measured against an opponent it can't
have influenced by being weak, so "it's winning more" can't just mean
"it's gotten better at losing to a worse version of itself."

Why check against an exhaustive minimax solver instead of, say, the
network's own previous checkpoint (the "arena" step real AlphaZero uses)?
Because tic-tac-toe is small enough to solve outright — there's a
computable, unambiguous notion of "playing perfectly" to measure against,
which most real games don't have. It also gives an evaluation invariant
that's satisfying to say out loud: a perfect player never loses, so **the
network's win count against it is always exactly zero**, by construction,
and the only number that can move is how often it manages to draw instead
of losing.

Run for real, this converges:

```
iteration 40: buffer=8452 loss=1.8831
eval vs minimax: wins=0 draws=38 losses=2
iteration 50: buffer=9847 loss=1.6104
eval vs minimax: wins=0 draws=40 losses=0
iteration 60: buffer=10000 loss=1.5721
eval vs minimax: wins=0 draws=40 losses=0
   ⋮                                    (draws=40 losses=0 holds through iteration 120)
```
*from a real `./train 120` run*

Loss falling and losses (the game-outcome kind) hitting zero are two
different claims, and it's worth noticing which one actually matters: a
network can have low loss while still losing occasionally, if it's
confidently predicting outcomes it's actually still misjudging
tactically. The number that ultimately validates this whole pipeline is
`losses=0` holding — not the loss curve.

---

## 07 — Concept map

A quick index back into the repository, for whichever fundamental you
want to see again in situ.

| Concept | Where it lives |
|---|---|
| Markov decision process (state / action / reward) | `az::Board`, `az::Cell`, `az::Outcome` — `include/az/board.hpp` |
| State canonicalization | Player-relative `encode()` — `src/board.cpp:62` |
| Policy π(a\|s) and value V(s) function approximation | `Network::predict`, the two-headed MLP — `src/network.cpp:23` |
| Planning / lookahead | `MCTS::simulate`, `MCTS::run` — `src/mcts.cpp:50, 87` |
| Exploration vs. exploitation (search-time) | The PUCT formula in `MCTS::selectChild` — `src/mcts.cpp:28` |
| Exploration vs. exploitation (trajectory-time) | `SelfPlayConfig::temperatureMoves`, sampling in `MCTS::run` — `src/selfplay.cpp`, `src/mcts.cpp:112` |
| Policy improvement operator | The MCTS visit distribution used as a training target for the policy head — `src/selfplay.cpp:24` |
| Monte Carlo return / credit assignment | The `z` label backfilled once a self-play game ends — `src/selfplay.cpp:28` |
| Combined loss / gradient descent | `Network::trainStep`, hand-derived backprop — `src/network.cpp:60` |
| Bias-free evaluation | `evaluateAgainstMinimax` against an exhaustive solver, never the network's own history — `src/eval.cpp`, `src/minimax.cpp` |

---

Written against commit-level source in this repository — build and run it
yourself with:

```sh
mkdir build && cd build && cmake .. && cmake --build .
./train 300 checkpoint.bin
```

Design and prose assembled with [Claude Code](https://claude.com/claude-code).
