# AlphaZero-Style Tic-Tac-Toe Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch, dependency-free C++ implementation of AlphaZero (self-play + PUCT/MCTS guided by a neural network) for tic-tac-toe, with an interactive CLI and a minimax-based convergence evaluator.

**Architecture:** A shared static library `az` (board, minimax, network, MCTS, replay buffer, self-play, eval helper), consumed by three executables (`train`, `evaluate`, `play_cli`). No external dependencies — the network's forward/backward pass is hand-written.

**Tech Stack:** C++17, CMake, C++ standard library only (no third-party deps, no test framework — plain `assert`-based test executables run via `ctest`).

**Spec:** `docs/superpowers/specs/2026-09-09-alphazero-tictactoe-design.md`

## Global Constraints

- C++17, CMake build, zero external dependencies (from the spec's Architecture/Build sections).
- Board encoding is always player-relative: 18 floats = [my stones (9), opponent stones (9)] from the perspective of `playerToMove()` (spec: `network` component).
- No arena/network-promotion gating — train continuously, evaluate periodically against minimax (spec: Rationale section).
- Tests are plain `assert`-based executables, no framework (spec: Testing section).
- Illegal moves are a programming error everywhere except `play_cli`'s human-input boundary, where they are rejected and re-prompted (spec: Error Handling section).

---

### Task 1: Project scaffolding + Board

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/az/board.hpp`
- Create: `src/board.cpp`
- Test: `tests/test_board.cpp`

**Interfaces:**
- Produces: `az::Cell` (enum: `Empty`, `X`, `O`), `az::Outcome` (enum: `Ongoing`, `XWins`, `OWins`, `Draw`), `az::Board` with `Board()`, `Cell cellAt(int) const`, `Cell playerToMove() const`, `bool isLegalMove(int) const`, `std::vector<int> legalMoves() const`, `Board applyMove(int) const`, `Outcome outcome() const`, `bool isTerminal() const`, `std::array<float,18> encode() const`, `bool operator==(const Board&) const`.

- [ ] **Step 1: Create the CMake project skeleton**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(alphazero_tictactoe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(az
    src/board.cpp
)
target_include_directories(az PUBLIC include)

enable_testing()

add_executable(test_board tests/test_board.cpp)
target_link_libraries(test_board az)
add_test(NAME test_board COMMAND test_board)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_board.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "az/board.hpp"

using namespace az;

void test_new_board_has_nine_legal_moves() {
    Board b;
    assert(b.legalMoves().size() == 9);
    assert(b.playerToMove() == Cell::X);
    assert(!b.isTerminal());
}

void test_apply_move_alternates_player() {
    Board b;
    Board b2 = b.applyMove(0);
    assert(b2.cellAt(0) == Cell::X);
    assert(b2.playerToMove() == Cell::O);
    assert(b2.legalMoves().size() == 8);
}

void test_row_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X
    b = b.applyMove(4); // O
    b = b.applyMove(2); // X completes top row
    assert(b.outcome() == Outcome::XWins);
    assert(b.isTerminal());
    assert(b.legalMoves().empty());
}

void test_diagonal_win_detected() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(1); // O
    b = b.applyMove(4); // X
    b = b.applyMove(2); // O
    b = b.applyMove(8); // X completes diagonal
    assert(b.outcome() == Outcome::XWins);
}

void test_draw_detected() {
    Board b;
    int moves[] = {0, 1, 2, 4, 3, 5, 7, 6, 8};
    for (int m : moves) b = b.applyMove(m);
    assert(b.outcome() == Outcome::Draw);
}

void test_illegal_move_rejected_by_isLegalMove() {
    Board b;
    b = b.applyMove(0);
    assert(!b.isLegalMove(0));
    assert(b.isLegalMove(1));
}

void test_encode_is_from_perspective_of_player_to_move() {
    Board b;
    b = b.applyMove(0); // X at 0, O to move
    auto enc = b.encode();
    for (int i = 0; i < 9; ++i) assert(enc[i] == 0.0f);
    assert(enc[9 + 0] == 1.0f);
}

int main() {
    test_new_board_has_nine_legal_moves();
    test_apply_move_alternates_player();
    test_row_win_detected();
    test_diagonal_win_detected();
    test_draw_detected();
    test_illegal_move_rejected_by_isLegalMove();
    test_encode_is_from_perspective_of_player_to_move();
    std::printf("all board tests passed\n");
    return 0;
}
```

- [ ] **Step 3: Run and verify it fails to build (board.hpp doesn't exist yet)**

Run: `mkdir -p build && cd build && cmake .. && cmake --build . --target test_board`
Expected: FAIL — `az/board.hpp` not found.

- [ ] **Step 4: Write `include/az/board.hpp`**

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace az {

enum class Cell : int8_t { Empty = 0, X = 1, O = 2 };

enum class Outcome { Ongoing, XWins, OWins, Draw };

class Board {
public:
    Board();

    Cell cellAt(int index) const;
    Cell playerToMove() const;

    bool isLegalMove(int index) const;
    std::vector<int> legalMoves() const;

    // Precondition: isLegalMove(index) == true (asserted).
    Board applyMove(int index) const;

    Outcome outcome() const;
    bool isTerminal() const { return outcome() != Outcome::Ongoing; }

    // 18 floats: [my stones (9), opponent stones (9)], from the
    // perspective of playerToMove().
    std::array<float, 18> encode() const;

    bool operator==(const Board& other) const;

private:
    std::array<Cell, 9> cells_;
    Cell toMove_;
};

} // namespace az
```

- [ ] **Step 5: Write `src/board.cpp`**

```cpp
#include "az/board.hpp"
#include <cassert>

namespace az {

namespace {
constexpr int kLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6},
};
}

Board::Board() : toMove_(Cell::X) {
    cells_.fill(Cell::Empty);
}

Cell Board::cellAt(int index) const {
    return cells_[index];
}

Cell Board::playerToMove() const {
    return toMove_;
}

bool Board::isLegalMove(int index) const {
    if (index < 0 || index >= 9) return false;
    if (isTerminal()) return false;
    return cells_[index] == Cell::Empty;
}

std::vector<int> Board::legalMoves() const {
    std::vector<int> moves;
    if (isTerminal()) return moves;
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) moves.push_back(i);
    }
    return moves;
}

Board Board::applyMove(int index) const {
    assert(isLegalMove(index));
    Board next = *this;
    next.cells_[index] = toMove_;
    next.toMove_ = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    return next;
}

Outcome Board::outcome() const {
    for (const auto& line : kLines) {
        Cell a = cells_[line[0]], b = cells_[line[1]], c = cells_[line[2]];
        if (a != Cell::Empty && a == b && b == c) {
            return a == Cell::X ? Outcome::XWins : Outcome::OWins;
        }
    }
    for (int i = 0; i < 9; ++i) {
        if (cells_[i] == Cell::Empty) return Outcome::Ongoing;
    }
    return Outcome::Draw;
}

std::array<float, 18> Board::encode() const {
    std::array<float, 18> out{};
    Cell mine = toMove_;
    Cell theirs = (toMove_ == Cell::X) ? Cell::O : Cell::X;
    for (int i = 0; i < 9; ++i) {
        out[i] = (cells_[i] == mine) ? 1.0f : 0.0f;
        out[9 + i] = (cells_[i] == theirs) ? 1.0f : 0.0f;
    }
    return out;
}

bool Board::operator==(const Board& other) const {
    return cells_ == other.cells_ && toMove_ == other.toMove_;
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_board && ctest -R test_board --output-on-failure`
Expected: PASS — `all board tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/board.hpp src/board.cpp tests/test_board.cpp
git commit -m "$(cat <<'EOF'
Add project scaffolding and Board component

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Minimax solver

**Files:**
- Create: `include/az/minimax.hpp`
- Create: `src/minimax.cpp`
- Test: `tests/test_minimax.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Board` (Task 1) — `isTerminal()`, `outcome()`, `legalMoves()`, `applyMove()`.
- Produces: `int az::minimaxBestMove(const Board&)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_minimax.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "az/board.hpp"
#include "az/minimax.hpp"

using namespace az;

void test_minimax_takes_immediate_win() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_blocks_immediate_loss() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(5); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2 -- O to move must block
    assert(minimaxBestMove(b) == 2);
}

void test_minimax_never_loses_against_itself() {
    Board b;
    while (!b.isTerminal()) {
        b = b.applyMove(minimaxBestMove(b));
    }
    assert(b.outcome() == Outcome::Draw);
}

int main() {
    test_minimax_takes_immediate_win();
    test_minimax_blocks_immediate_loss();
    test_minimax_never_loses_against_itself();
    std::printf("all minimax tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/minimax.cpp` to the `az` library sources, and append:

```cmake
add_executable(test_minimax tests/test_minimax.cpp)
target_link_libraries(test_minimax az)
add_test(NAME test_minimax COMMAND test_minimax)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_minimax`
Expected: FAIL — `az/minimax.hpp` not found.

- [ ] **Step 4: Write `include/az/minimax.hpp`**

```cpp
#pragma once
#include "az/board.hpp"

namespace az {

// Returns the game-theoretically optimal move for board.playerToMove(),
// via exhaustive minimax search. Precondition: !board.isTerminal().
int minimaxBestMove(const Board& board);

} // namespace az
```

- [ ] **Step 5: Write `src/minimax.cpp`**

```cpp
#include "az/minimax.hpp"
#include <cassert>
#include <limits>

namespace az {

namespace {

// Value of `board` from the perspective of board.playerToMove(), assuming
// optimal play by both sides: +1 win, -1 loss, 0 draw.
int scoreOf(const Board& board) {
    if (board.isTerminal()) {
        return board.outcome() == Outcome::Draw ? 0 : -1;
    }
    int best = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int childScore = -scoreOf(board.applyMove(m));
        if (childScore > best) best = childScore;
    }
    return best;
}

} // namespace

int minimaxBestMove(const Board& board) {
    assert(!board.isTerminal());
    int bestMove = board.legalMoves().front();
    int bestScore = std::numeric_limits<int>::min();
    for (int m : board.legalMoves()) {
        int score = -scoreOf(board.applyMove(m));
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    return bestMove;
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_minimax && ctest -R test_minimax --output-on-failure`
Expected: PASS — `all minimax tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/minimax.hpp src/minimax.cpp tests/test_minimax.cpp
git commit -m "$(cat <<'EOF'
Add exhaustive minimax solver

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Neural network (forward + backprop + save/load)

**Files:**
- Create: `include/az/network.hpp`
- Create: `src/network.cpp`
- Test: `tests/test_network.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `az::Prediction{policy: array<float,9>, value: float}`, `az::TrainingExample{encodedBoard: array<float,18>, targetPolicy: array<float,9>, targetValue: float}`, `az::Network` with `Network()`, `Prediction predict(const std::array<float,18>&) const`, `float trainStep(const std::vector<TrainingExample>&, float learningRate)`, `void save(const std::string&) const`, `void load(const std::string&)`. Constants `Network::kInputSize=18`, `kHiddenSize=64`, `kPolicySize=9`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_network.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "az/network.hpp"

using namespace az;

void test_predict_output_shapes_and_ranges() {
    Network net;
    std::array<float, 18> input{};
    input[0] = 1.0f;
    Prediction pred = net.predict(input);
    float sum = 0.0f;
    for (float p : pred.policy) {
        assert(p >= 0.0f);
        sum += p;
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);
    assert(pred.value >= -1.0f && pred.value <= 1.0f);
}

void test_train_step_reduces_loss_on_fixed_batch() {
    Network net;
    TrainingExample example;
    example.encodedBoard.fill(0.0f);
    example.encodedBoard[0] = 1.0f;
    example.targetPolicy.fill(0.0f);
    example.targetPolicy[4] = 1.0f;
    example.targetValue = 1.0f;

    std::vector<TrainingExample> batch{example};

    float firstLoss = net.trainStep(batch, 0.05f);
    float lastLoss = firstLoss;
    for (int i = 0; i < 200; ++i) {
        lastLoss = net.trainStep(batch, 0.05f);
    }
    assert(lastLoss < firstLoss);
    assert(lastLoss < 0.1f);
}

int main() {
    test_predict_output_shapes_and_ranges();
    test_train_step_reduces_loss_on_fixed_batch();
    std::printf("all network tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/network.cpp` to `az` library sources, and append:

```cmake
add_executable(test_network tests/test_network.cpp)
target_link_libraries(test_network az)
add_test(NAME test_network COMMAND test_network)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_network`
Expected: FAIL — `az/network.hpp` not found.

- [ ] **Step 4: Write `include/az/network.hpp`**

```cpp
#pragma once
#include <array>
#include <string>
#include <vector>

namespace az {

struct Prediction {
    std::array<float, 9> policy;
    float value;
};

struct TrainingExample {
    std::array<float, 18> encodedBoard;
    std::array<float, 9> targetPolicy;
    float targetValue;
};

class Network {
public:
    static constexpr int kInputSize = 18;
    static constexpr int kHiddenSize = 64;
    static constexpr int kPolicySize = 9;

    Network();

    // Full softmax over all 9 outputs -- NOT masked to legal moves. Callers
    // with board context (MCTS) mask and renormalize themselves.
    Prediction predict(const std::array<float, kInputSize>& encodedBoard) const;

    // One gradient-descent step over a batch. Returns the batch's mean
    // combined loss (value MSE + policy cross-entropy).
    float trainStep(const std::vector<TrainingExample>& batch, float learningRate);

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::array<std::array<float, kInputSize>, kHiddenSize> w1_;
    std::array<float, kHiddenSize> b1_;
    std::array<std::array<float, kHiddenSize>, kPolicySize> wPolicy_;
    std::array<float, kPolicySize> bPolicy_;
    std::array<float, kHiddenSize> wValue_;
    float bValue_;
};

} // namespace az
```

- [ ] **Step 5: Write `src/network.cpp`**

```cpp
#include "az/network.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace az {

Network::Network() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& row : w1_) for (float& w : row) w = dist(gen);
    b1_.fill(0.0f);
    for (auto& row : wPolicy_) for (float& w : row) w = dist(gen);
    bPolicy_.fill(0.0f);
    for (float& w : wValue_) w = dist(gen);
    bValue_ = 0.0f;
}

Prediction Network::predict(const std::array<float, kInputSize>& encodedBoard) const {
    std::array<float, kHiddenSize> a1{};
    for (int h = 0; h < kHiddenSize; ++h) {
        float z = b1_[h];
        for (int i = 0; i < kInputSize; ++i) z += w1_[h][i] * encodedBoard[i];
        a1[h] = z > 0.0f ? z : 0.0f;
    }

    std::array<float, kPolicySize> logits{};
    for (int k = 0; k < kPolicySize; ++k) {
        float z = bPolicy_[k];
        for (int h = 0; h < kHiddenSize; ++h) z += wPolicy_[k][h] * a1[h];
        logits[k] = z;
    }
    float maxLogit = logits[0];
    for (float l : logits) maxLogit = std::max(maxLogit, l);
    float sumExp = 0.0f;
    std::array<float, kPolicySize> policy{};
    for (int k = 0; k < kPolicySize; ++k) {
        policy[k] = std::exp(logits[k] - maxLogit);
        sumExp += policy[k];
    }
    for (float& p : policy) p /= sumExp;

    float valuePre = bValue_;
    for (int h = 0; h < kHiddenSize; ++h) valuePre += wValue_[h] * a1[h];
    float value = std::tanh(valuePre);

    return Prediction{policy, value};
}

float Network::trainStep(const std::vector<TrainingExample>& batch, float learningRate) {
    std::array<std::array<float, kInputSize>, kHiddenSize> gradW1{};
    std::array<float, kHiddenSize> gradB1{};
    std::array<std::array<float, kHiddenSize>, kPolicySize> gradWPolicy{};
    std::array<float, kPolicySize> gradBPolicy{};
    std::array<float, kHiddenSize> gradWValue{};
    float gradBValue = 0.0f;

    float totalLoss = 0.0f;

    for (const auto& example : batch) {
        std::array<float, kHiddenSize> z1{}, a1{};
        for (int h = 0; h < kHiddenSize; ++h) {
            float z = b1_[h];
            for (int i = 0; i < kInputSize; ++i) z += w1_[h][i] * example.encodedBoard[i];
            z1[h] = z;
            a1[h] = z > 0.0f ? z : 0.0f;
        }

        std::array<float, kPolicySize> logits{};
        for (int k = 0; k < kPolicySize; ++k) {
            float z = bPolicy_[k];
            for (int h = 0; h < kHiddenSize; ++h) z += wPolicy_[k][h] * a1[h];
            logits[k] = z;
        }
        float maxLogit = logits[0];
        for (float l : logits) maxLogit = std::max(maxLogit, l);
        float sumExp = 0.0f;
        std::array<float, kPolicySize> policy{};
        for (int k = 0; k < kPolicySize; ++k) {
            policy[k] = std::exp(logits[k] - maxLogit);
            sumExp += policy[k];
        }
        for (float& p : policy) p /= sumExp;

        float valuePre = bValue_;
        for (int h = 0; h < kHiddenSize; ++h) valuePre += wValue_[h] * a1[h];
        float value = std::tanh(valuePre);

        float valueLoss = (value - example.targetValue) * (value - example.targetValue);
        float policyLoss = 0.0f;
        for (int k = 0; k < kPolicySize; ++k) {
            policyLoss -= example.targetPolicy[k] * std::log(policy[k] + 1e-8f);
        }
        totalLoss += valueLoss + policyLoss;

        float dValuePre = 2.0f * (value - example.targetValue) * (1.0f - value * value);

        std::array<float, kPolicySize> dLogits{};
        for (int k = 0; k < kPolicySize; ++k) dLogits[k] = policy[k] - example.targetPolicy[k];

        std::array<float, kHiddenSize> dA1{};
        for (int h = 0; h < kHiddenSize; ++h) {
            float grad = dValuePre * wValue_[h];
            for (int k = 0; k < kPolicySize; ++k) grad += dLogits[k] * wPolicy_[k][h];
            dA1[h] = grad;
        }
        std::array<float, kHiddenSize> dZ1{};
        for (int h = 0; h < kHiddenSize; ++h) dZ1[h] = z1[h] > 0.0f ? dA1[h] : 0.0f;

        for (int h = 0; h < kHiddenSize; ++h) {
            for (int i = 0; i < kInputSize; ++i) gradW1[h][i] += dZ1[h] * example.encodedBoard[i];
            gradB1[h] += dZ1[h];
        }
        for (int k = 0; k < kPolicySize; ++k) {
            for (int h = 0; h < kHiddenSize; ++h) gradWPolicy[k][h] += dLogits[k] * a1[h];
            gradBPolicy[k] += dLogits[k];
        }
        for (int h = 0; h < kHiddenSize; ++h) gradWValue[h] += dValuePre * a1[h];
        gradBValue += dValuePre;
    }

    float n = static_cast<float>(batch.size());
    for (int h = 0; h < kHiddenSize; ++h) {
        for (int i = 0; i < kInputSize; ++i) w1_[h][i] -= learningRate * gradW1[h][i] / n;
        b1_[h] -= learningRate * gradB1[h] / n;
    }
    for (int k = 0; k < kPolicySize; ++k) {
        for (int h = 0; h < kHiddenSize; ++h) wPolicy_[k][h] -= learningRate * gradWPolicy[k][h] / n;
        bPolicy_[k] -= learningRate * gradBPolicy[k] / n;
    }
    for (int h = 0; h < kHiddenSize; ++h) wValue_[h] -= learningRate * gradWValue[h] / n;
    bValue_ -= learningRate * gradBValue / n;

    return totalLoss / n;
}

void Network::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Network::save: cannot open " + path);
    for (const auto& row : w1_) out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(b1_.data()), b1_.size() * sizeof(float));
    for (const auto& row : wPolicy_) out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(bPolicy_.data()), bPolicy_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(wValue_.data()), wValue_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(&bValue_), sizeof(float));
}

void Network::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Network::load: cannot open " + path);
    for (auto& row : w1_) in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(b1_.data()), b1_.size() * sizeof(float));
    for (auto& row : wPolicy_) in.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(bPolicy_.data()), bPolicy_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(wValue_.data()), wValue_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(&bValue_), sizeof(float));
    if (!in) throw std::runtime_error("Network::load: truncated file " + path);
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_network && ctest -R test_network --output-on-failure`
Expected: PASS — `all network tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/network.hpp src/network.cpp tests/test_network.cpp
git commit -m "$(cat <<'EOF'
Add hand-written neural network (forward, backprop, checkpointing)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: MCTS (PUCT search)

**Files:**
- Create: `include/az/mcts.hpp`
- Create: `src/mcts.cpp`
- Test: `tests/test_mcts.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Board` (Task 1), `az::Network`/`Prediction` (Task 3).
- Produces: `az::MCTSResult{visitDistribution: array<float,9>, selectedMove: int}`, `az::MCTS` with `MCTS(const Network&, int numSimulations, float cPuct=1.5f)`, `MCTSResult run(const Board&, float temperature)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcts.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "az/board.hpp"
#include "az/mcts.hpp"
#include "az/network.hpp"

using namespace az;

void test_mcts_finds_immediate_winning_move() {
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O

    Network net;
    MCTS mcts(net, /*numSimulations=*/300, /*cPuct=*/1.5f);
    MCTSResult result = mcts.run(b, /*temperature=*/0.0f);

    assert(result.selectedMove == 2);
    for (int m : b.legalMoves()) {
        if (m != 2) assert(result.visitDistribution[2] > result.visitDistribution[m]);
    }
}

int main() {
    test_mcts_finds_immediate_winning_move();
    std::printf("all mcts tests passed\n");
    return 0;
}
```

Note: this test uses a freshly (randomly) initialized `Network`, not a trained one. It's expected to pass regardless of the network's quality: the winning move's child is a terminal node, so its true value (+1) gets backed up on its very first visit, immediately dominating PUCT selection over untrained/noisy priors.

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/mcts.cpp` to `az` library sources, and append:

```cmake
add_executable(test_mcts tests/test_mcts.cpp)
target_link_libraries(test_mcts az)
add_test(NAME test_mcts COMMAND test_mcts)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_mcts`
Expected: FAIL — `az/mcts.hpp` not found.

- [ ] **Step 4: Write `include/az/mcts.hpp`**

```cpp
#pragma once
#include <array>
#include <memory>
#include <random>
#include "az/board.hpp"
#include "az/network.hpp"

namespace az {

struct MCTSResult {
    std::array<float, 9> visitDistribution;
    int selectedMove;
};

class MCTS {
public:
    MCTS(const Network& network, int numSimulations, float cPuct = 1.5f);

    // temperature > 0: sample proportional to visitCount^(1/temperature).
    // temperature == 0: pick the max-visit move (ties -> lowest index).
    MCTSResult run(const Board& board, float temperature);

private:
    struct Node {
        Board board;
        bool expanded = false;
        std::array<float, 9> priors{};
        std::array<int, 9> visitCounts{};
        std::array<float, 9> totalValue{};
        std::array<std::unique_ptr<Node>, 9> children{};
    };

    float expand(Node& node);
    float simulate(Node& node);
    int selectChild(const Node& node) const;

    const Network& network_;
    int numSimulations_;
    float cPuct_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace az
```

- [ ] **Step 5: Write `src/mcts.cpp`**

```cpp
#include "az/mcts.hpp"
#include <cmath>

namespace az {

MCTS::MCTS(const Network& network, int numSimulations, float cPuct)
    : network_(network), numSimulations_(numSimulations), cPuct_(cPuct) {}

float MCTS::expand(Node& node) {
    Prediction pred = network_.predict(node.board.encode());
    std::array<float, 9> masked{};
    float sum = 0.0f;
    for (int m : node.board.legalMoves()) {
        masked[m] = pred.policy[m];
        sum += masked[m];
    }
    if (sum > 1e-8f) {
        for (int m : node.board.legalMoves()) masked[m] /= sum;
    } else {
        float uniform = 1.0f / static_cast<float>(node.board.legalMoves().size());
        for (int m : node.board.legalMoves()) masked[m] = uniform;
    }
    node.priors = masked;
    node.expanded = true;
    return pred.value;
}

int MCTS::selectChild(const Node& node) const {
    int totalVisits = 0;
    for (int m : node.board.legalMoves()) totalVisits += node.visitCounts[m];

    float bestScore = -1e9f;
    int bestMove = node.board.legalMoves().front();
    for (int m : node.board.legalMoves()) {
        float q = node.visitCounts[m] > 0 ? node.totalValue[m] / node.visitCounts[m] : 0.0f;
        float u = cPuct_ * node.priors[m] * std::sqrt(static_cast<float>(totalVisits) + 1e-8f)
                  / (1 + node.visitCounts[m]);
        float score = q + u;
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    return bestMove;
}

float MCTS::simulate(Node& node) {
    if (node.board.isTerminal()) {
        return node.board.outcome() == Outcome::Draw ? 0.0f : -1.0f;
    }
    if (!node.expanded) {
        return expand(node);
    }
    int move = selectChild(node);
    if (!node.children[move]) {
        node.children[move] = std::make_unique<Node>();
        node.children[move]->board = node.board.applyMove(move);
    }
    float childValue = simulate(*node.children[move]);
    float value = -childValue;
    node.visitCounts[move] += 1;
    node.totalValue[move] += value;
    return value;
}

MCTSResult MCTS::run(const Board& board, float temperature) {
    Node root;
    root.board = board;
    for (int i = 0; i < numSimulations_; ++i) {
        simulate(root);
    }

    std::array<float, 9> dist{};
    int totalVisits = 0;
    for (int m : board.legalMoves()) totalVisits += root.visitCounts[m];
    if (totalVisits > 0) {
        for (int m : board.legalMoves()) {
            dist[m] = static_cast<float>(root.visitCounts[m]) / static_cast<float>(totalVisits);
        }
    } else {
        float uniform = 1.0f / static_cast<float>(board.legalMoves().size());
        for (int m : board.legalMoves()) dist[m] = uniform;
    }

    int selected;
    if (temperature <= 0.0f) {
        selected = board.legalMoves().front();
        int bestVisits = -1;
        for (int m : board.legalMoves()) {
            if (root.visitCounts[m] > bestVisits) {
                bestVisits = root.visitCounts[m];
                selected = m;
            }
        }
    } else {
        std::array<float, 9> weights{};
        float sum = 0.0f;
        for (int m : board.legalMoves()) {
            weights[m] = std::pow(static_cast<float>(root.visitCounts[m]), 1.0f / temperature);
            sum += weights[m];
        }
        std::uniform_real_distribution<float> unif(0.0f, sum);
        float r = unif(rng_);
        float cumulative = 0.0f;
        selected = board.legalMoves().back();
        for (int m : board.legalMoves()) {
            cumulative += weights[m];
            if (r <= cumulative) {
                selected = m;
                break;
            }
        }
    }

    return MCTSResult{dist, selected};
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_mcts && ctest -R test_mcts --output-on-failure`
Expected: PASS — `all mcts tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/mcts.hpp src/mcts.cpp tests/test_mcts.cpp
git commit -m "$(cat <<'EOF'
Add PUCT/MCTS search guided by the network

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Replay buffer

**Files:**
- Create: `include/az/replay_buffer.hpp`
- Create: `src/replay_buffer.cpp`
- Test: `tests/test_replay_buffer.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::TrainingExample` (Task 3).
- Produces: `az::ReplayBuffer` with `explicit ReplayBuffer(size_t capacity)`, `void add(const TrainingExample&)`, `std::vector<TrainingExample> sampleBatch(size_t batchSize) const`, `size_t size() const`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_buffer.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "az/replay_buffer.hpp"

using namespace az;

namespace {
TrainingExample makeExample(float value) {
    TrainingExample ex;
    ex.encodedBoard.fill(0.0f);
    ex.targetPolicy.fill(0.0f);
    ex.targetValue = value;
    return ex;
}
}

void test_add_and_size() {
    ReplayBuffer buf(3);
    assert(buf.size() == 0);
    buf.add(makeExample(1.0f));
    assert(buf.size() == 1);
}

void test_capacity_overwrites_oldest() {
    ReplayBuffer buf(2);
    buf.add(makeExample(1.0f));
    buf.add(makeExample(2.0f));
    buf.add(makeExample(3.0f)); // overwrites the value=1.0 example
    assert(buf.size() == 2);
    auto batch = buf.sampleBatch(50);
    bool sawOne = false;
    for (const auto& ex : batch) if (ex.targetValue == 1.0f) sawOne = true;
    assert(!sawOne);
}

void test_sample_batch_returns_requested_size() {
    ReplayBuffer buf(5);
    buf.add(makeExample(1.0f));
    buf.add(makeExample(2.0f));
    auto batch = buf.sampleBatch(10);
    assert(batch.size() == 10);
}

int main() {
    test_add_and_size();
    test_capacity_overwrites_oldest();
    test_sample_batch_returns_requested_size();
    std::printf("all replay_buffer tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/replay_buffer.cpp` to `az` library sources, and append:

```cmake
add_executable(test_replay_buffer tests/test_replay_buffer.cpp)
target_link_libraries(test_replay_buffer az)
add_test(NAME test_replay_buffer COMMAND test_replay_buffer)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_replay_buffer`
Expected: FAIL — `az/replay_buffer.hpp` not found.

- [ ] **Step 4: Write `include/az/replay_buffer.hpp`**

```cpp
#pragma once
#include <random>
#include <vector>
#include "az/network.hpp"

namespace az {

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void add(const TrainingExample& example);

    // Uniform random sampling with replacement. Precondition: size() > 0.
    std::vector<TrainingExample> sampleBatch(size_t batchSize) const;

    size_t size() const { return buffer_.size(); }

private:
    size_t capacity_;
    size_t nextIndex_ = 0;
    std::vector<TrainingExample> buffer_;
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace az
```

- [ ] **Step 5: Write `src/replay_buffer.cpp`**

```cpp
#include "az/replay_buffer.hpp"
#include <cassert>

namespace az {

ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity) {
    buffer_.reserve(capacity);
}

void ReplayBuffer::add(const TrainingExample& example) {
    if (buffer_.size() < capacity_) {
        buffer_.push_back(example);
    } else {
        buffer_[nextIndex_] = example;
    }
    nextIndex_ = (nextIndex_ + 1) % capacity_;
}

std::vector<TrainingExample> ReplayBuffer::sampleBatch(size_t batchSize) const {
    assert(!buffer_.empty());
    std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
    std::vector<TrainingExample> batch;
    batch.reserve(batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        batch.push_back(buffer_[dist(rng_)]);
    }
    return batch;
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_replay_buffer && ctest -R test_replay_buffer --output-on-failure`
Expected: PASS — `all replay_buffer tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/replay_buffer.hpp src/replay_buffer.cpp tests/test_replay_buffer.cpp
git commit -m "$(cat <<'EOF'
Add fixed-capacity replay buffer

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Self-play

**Files:**
- Create: `include/az/selfplay.hpp`
- Create: `src/selfplay.cpp`
- Test: `tests/test_selfplay.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Board` (Task 1), `az::Network`/`TrainingExample` (Task 3), `az::MCTS`/`MCTSResult` (Task 4), `az::ReplayBuffer` (Task 5).
- Produces: `az::SelfPlayConfig{numSimulations=50, cPuct=1.5f, temperatureMoves=2}`, `void az::playSelfPlayGame(const Network&, const SelfPlayConfig&, ReplayBuffer&)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_selfplay.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include "az/network.hpp"
#include "az/replay_buffer.hpp"
#include "az/selfplay.hpp"

using namespace az;

void test_selfplay_game_populates_buffer() {
    Network net;
    ReplayBuffer buffer(100);
    SelfPlayConfig config;
    config.numSimulations = 20;

    playSelfPlayGame(net, config, buffer);

    assert(buffer.size() >= 5 && buffer.size() <= 9);

    auto batch = buffer.sampleBatch(buffer.size());
    for (const auto& ex : batch) {
        float sum = 0.0f;
        for (float p : ex.targetPolicy) {
            assert(p >= 0.0f);
            sum += p;
        }
        assert(std::fabs(sum - 1.0f) < 1e-3f);
        assert(ex.targetValue == 1.0f || ex.targetValue == -1.0f || ex.targetValue == 0.0f);
    }
}

int main() {
    test_selfplay_game_populates_buffer();
    std::printf("all selfplay tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/selfplay.cpp` to `az` library sources, and append:

```cmake
add_executable(test_selfplay tests/test_selfplay.cpp)
target_link_libraries(test_selfplay az)
add_test(NAME test_selfplay COMMAND test_selfplay)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_selfplay`
Expected: FAIL — `az/selfplay.hpp` not found.

- [ ] **Step 4: Write `include/az/selfplay.hpp`**

```cpp
#pragma once
#include "az/network.hpp"
#include "az/replay_buffer.hpp"

namespace az {

struct SelfPlayConfig {
    int numSimulations = 50;
    float cPuct = 1.5f;
    // Number of plies (from game start) using temperature=1.0 sampling;
    // afterward, moves are selected greedily (temperature=0).
    int temperatureMoves = 2;
};

// Plays one full self-play game guided by `network`+MCTS, pushing every
// position's (encodedBoard, mctsPolicy, outcome) into `buffer`.
void playSelfPlayGame(const Network& network, const SelfPlayConfig& config, ReplayBuffer& buffer);

} // namespace az
```

- [ ] **Step 5: Write `src/selfplay.cpp`**

```cpp
#include "az/selfplay.hpp"
#include <vector>
#include "az/board.hpp"
#include "az/mcts.hpp"

namespace az {

namespace {
struct PendingExample {
    std::array<float, 18> encoded;
    std::array<float, 9> policy;
    Cell playerToMove;
};
}

void playSelfPlayGame(const Network& network, const SelfPlayConfig& config, ReplayBuffer& buffer) {
    Board board;
    std::vector<PendingExample> pending;

    int ply = 0;
    while (!board.isTerminal()) {
        MCTS mcts(network, config.numSimulations, config.cPuct);
        float temperature = (ply < config.temperatureMoves) ? 1.0f : 0.0f;
        MCTSResult result = mcts.run(board, temperature);
        pending.push_back(PendingExample{board.encode(), result.visitDistribution, board.playerToMove()});
        board = board.applyMove(result.selectedMove);
        ++ply;
    }

    Outcome outcome = board.outcome();
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
}

} // namespace az
```

- [ ] **Step 6: Build and run the test**

Run: `cd build && cmake --build . --target test_selfplay && ctest -R test_selfplay --output-on-failure`
Expected: PASS — `all selfplay tests passed`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/az/selfplay.hpp src/selfplay.cpp tests/test_selfplay.cpp
git commit -m "$(cat <<'EOF'
Add self-play game generation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Minimax-eval helper + `evaluate` executable

**Files:**
- Create: `include/az/eval.hpp`
- Create: `src/eval.cpp`
- Create: `tests/test_eval.cpp`
- Create: `apps/evaluate.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Board` (Task 1), `az::minimaxBestMove` (Task 2), `az::Network` (Task 3), `az::MCTS` (Task 4).
- Produces: `az::EvalResult{wins=0, draws=0, losses=0}`, `az::EvalResult az::evaluateAgainstMinimax(const Network&, int gamesPerSide, int numSimulations)` — results always from the network's perspective.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include "az/eval.hpp"
#include "az/network.hpp"

using namespace az;

void test_eval_result_totals_match_games_played() {
    Network net;
    EvalResult result = evaluateAgainstMinimax(net, /*gamesPerSide=*/3, /*numSimulations=*/20);
    assert(result.wins + result.draws + result.losses == 6);
    assert(result.wins == 0); // perfect minimax never loses, so the network can never "win"
}

int main() {
    test_eval_result_totals_match_games_played();
    std::printf("all eval tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Add the CMake targets**

Edit `CMakeLists.txt`: add `src/eval.cpp` to `az` library sources, and append:

```cmake
add_executable(test_eval tests/test_eval.cpp)
target_link_libraries(test_eval az)
add_test(NAME test_eval COMMAND test_eval)

add_executable(evaluate apps/evaluate.cpp)
target_link_libraries(evaluate az)
```

- [ ] **Step 3: Run and verify it fails to build**

Run: `cd build && cmake .. && cmake --build . --target test_eval`
Expected: FAIL — `az/eval.hpp` not found.

- [ ] **Step 4: Write `include/az/eval.hpp`**

```cpp
#pragma once
#include "az/network.hpp"

namespace az {

struct EvalResult {
    int wins = 0;
    int draws = 0;
    int losses = 0;
};

// Plays gamesPerSide games with the network as X and gamesPerSide as O
// against perfect minimax, using greedy (temperature=0) MCTS with
// numSimulations simulations per move. Results are from the network's
// perspective.
EvalResult evaluateAgainstMinimax(const Network& network, int gamesPerSide, int numSimulations);

} // namespace az
```

- [ ] **Step 5: Write `src/eval.cpp`**

```cpp
#include "az/eval.hpp"
#include "az/board.hpp"
#include "az/mcts.hpp"
#include "az/minimax.hpp"

namespace az {

namespace {

int playOneGame(const Network& network, bool networkPlaysX, int numSimulations) {
    Board board;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            MCTS mcts(network, numSimulations, 1.5f);
            move = mcts.run(board, 0.0f).selectedMove;
        } else {
            move = minimaxBestMove(board);
        }
        board = board.applyMove(move);
    }
    Outcome o = board.outcome();
    if (o == Outcome::Draw) return 0;
    bool networkWon = (o == Outcome::XWins && networkPlaysX) || (o == Outcome::OWins && !networkPlaysX);
    return networkWon ? 1 : -1;
}

} // namespace

EvalResult evaluateAgainstMinimax(const Network& network, int gamesPerSide, int numSimulations) {
    EvalResult result;
    auto record = [&](int r) {
        if (r == 1) result.wins++;
        else if (r == -1) result.losses++;
        else result.draws++;
    };
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, true, numSimulations));
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, false, numSimulations));
    return result;
}

} // namespace az
```

- [ ] **Step 6: Write `apps/evaluate.cpp`**

```cpp
#include <cstdio>
#include <cstdlib>
#include <string>
#include "az/eval.hpp"
#include "az/network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: evaluate <checkpoint-path> [games-per-side]\n");
        return 1;
    }
    std::string checkpointPath = argv[1];
    int gamesPerSide = argc >= 3 ? std::atoi(argv[2]) : 50;

    az::Network network;
    network.load(checkpointPath);

    az::EvalResult result = az::evaluateAgainstMinimax(network, gamesPerSide, /*numSimulations=*/100);
    std::printf("vs minimax over %d games/side: wins=%d draws=%d losses=%d\n",
                gamesPerSide, result.wins, result.draws, result.losses);
    return 0;
}
```

- [ ] **Step 7: Build test_eval, run it, then build evaluate and smoke-test it**

Run: `cd build && cmake --build . --target test_eval evaluate && ctest -R test_eval --output-on-failure`
Expected: PASS — `all eval tests passed`.

Run:
```bash
cd build
cat > /tmp/save_checkpoint.cpp << 'CPPEOF'
#include "az/network.hpp"
int main() { az::Network net; net.save("/tmp/az_smoke_checkpoint.bin"); return 0; }
CPPEOF
g++ -std=c++17 -I../include /tmp/save_checkpoint.cpp ../src/network.cpp -o /tmp/save_checkpoint
/tmp/save_checkpoint
./evaluate /tmp/az_smoke_checkpoint.bin 3
```
Expected: a line like `vs minimax over 3 games/side: wins=0 draws=D losses=L` with `D + L == 6` and `wins=0`.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt include/az/eval.hpp src/eval.cpp tests/test_eval.cpp apps/evaluate.cpp
git commit -m "$(cat <<'EOF'
Add minimax-eval helper and evaluate executable

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `train` executable

**Files:**
- Create: `apps/train.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Network`, `az::ReplayBuffer`, `az::SelfPlayConfig`/`playSelfPlayGame` (Task 6), `az::evaluateAgainstMinimax` (Task 7).
- Produces: a `train` binary taking optional `argv[1]=numIterations`, `argv[2]=checkpointPath`; saves periodic checkpoints and prints iteration/eval progress to stdout.

- [ ] **Step 1: Write `apps/train.cpp`**

```cpp
#include <cstdio>
#include <cstdlib>
#include <string>
#include "az/eval.hpp"
#include "az/network.hpp"
#include "az/replay_buffer.hpp"
#include "az/selfplay.hpp"

int main(int argc, char** argv) {
    int numIterations = argc >= 2 ? std::atoi(argv[1]) : 200;
    std::string checkpointPath = argc >= 3 ? argv[2] : "checkpoint.bin";

    const int gamesPerIteration = 25;
    const int batchSize = 32;
    const int trainStepsPerIteration = 20;
    const float learningRate = 0.01f;
    const int evalEveryIterations = 10;
    const int evalGamesPerSide = 20;

    az::Network network;
    az::ReplayBuffer buffer(/*capacity=*/10000);
    az::SelfPlayConfig selfPlayConfig;

    for (int iter = 0; iter < numIterations; ++iter) {
        for (int g = 0; g < gamesPerIteration; ++g) {
            az::playSelfPlayGame(network, selfPlayConfig, buffer);
        }

        if (buffer.size() >= static_cast<size_t>(batchSize)) {
            float lastLoss = 0.0f;
            for (int s = 0; s < trainStepsPerIteration; ++s) {
                auto batch = buffer.sampleBatch(batchSize);
                lastLoss = network.trainStep(batch, learningRate);
            }
            std::printf("iteration %d: buffer=%zu loss=%.4f\n", iter, buffer.size(), lastLoss);
        }

        if ((iter + 1) % evalEveryIterations == 0) {
            az::EvalResult result = az::evaluateAgainstMinimax(network, evalGamesPerSide, /*numSimulations=*/100);
            std::printf("eval vs minimax: wins=%d draws=%d losses=%d\n", result.wins, result.draws, result.losses);
            network.save(checkpointPath);
            std::printf("checkpoint saved to %s\n", checkpointPath.c_str());
        }
    }

    network.save(checkpointPath);
    std::printf("final checkpoint saved to %s\n", checkpointPath.c_str());
    return 0;
}
```

- [ ] **Step 2: Add the CMake target**

Edit `CMakeLists.txt`: append:

```cmake
add_executable(train apps/train.cpp)
target_link_libraries(train az)
```

- [ ] **Step 3: Build and smoke-test with a tiny run**

Run: `cd build && cmake .. && cmake --build . --target train`
Expected: builds successfully.

Run: `cd build && ./train 3 /tmp/az_train_smoke.bin`
Expected: prints 3 `iteration N: ...` lines and exits 0 (no crash); `/tmp/az_train_smoke.bin` exists afterward (`ls -la /tmp/az_train_smoke.bin`).

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt apps/train.cpp
git commit -m "$(cat <<'EOF'
Add train executable orchestrating self-play + training + eval

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `play_cli` executable

**Files:**
- Create: `apps/play_cli.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `az::Board` (Task 1), `az::Network` (Task 3), `az::MCTS` (Task 4).
- Produces: a `play_cli` binary taking `argv[1]=checkpointPath`, running an interactive human-vs-agent game on stdin/stdout.

- [ ] **Step 1: Write `apps/play_cli.cpp`**

```cpp
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "az/board.hpp"
#include "az/mcts.hpp"
#include "az/network.hpp"

namespace {

void printBoard(const az::Board& board) {
    const char* symbols[3] = {".", "X", "O"};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            int idx = row * 3 + col;
            std::printf("%s ", symbols[static_cast<int>(board.cellAt(idx))]);
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: play_cli <checkpoint-path>\n");
        return 1;
    }

    az::Network network;
    network.load(argv[1]);

    std::printf("You are X. Enter a move as a number 0-8 (see grid below).\n");
    std::printf("0 1 2\n3 4 5\n6 7 8\n\n");

    az::Board board;
    while (!board.isTerminal()) {
        printBoard(board);
        if (board.playerToMove() == az::Cell::X) {
            int move = -1;
            while (true) {
                std::printf("Your move: ");
                if (!(std::cin >> move) || !board.isLegalMove(move)) {
                    std::printf("Invalid move, try again.\n");
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                    continue;
                }
                break;
            }
            board = board.applyMove(move);
        } else {
            az::MCTS mcts(network, /*numSimulations=*/200, /*cPuct=*/1.5f);
            int move = mcts.run(board, 0.0f).selectedMove;
            std::printf("Agent plays %d\n", move);
            board = board.applyMove(move);
        }
    }

    printBoard(board);
    az::Outcome outcome = board.outcome();
    if (outcome == az::Outcome::Draw) std::printf("Draw.\n");
    else if (outcome == az::Outcome::XWins) std::printf("You win!\n");
    else std::printf("Agent wins.\n");

    return 0;
}
```

- [ ] **Step 2: Add the CMake target**

Edit `CMakeLists.txt`: append:

```cmake
add_executable(play_cli apps/play_cli.cpp)
target_link_libraries(play_cli az)
```

- [ ] **Step 3: Build and smoke-test with piped input**

Run: `cd build && cmake .. && cmake --build . --target play_cli`
Expected: builds successfully.

Run: `cd build && echo "0 1 2 3 4 5 6 7 8" | ./play_cli /tmp/az_train_smoke.bin`
Expected: exits 0, prints the board after each move, and ends with one of `Draw.` / `You win!` / `Agent wins.` (uses the checkpoint produced in Task 8's smoke test).

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt apps/play_cli.cpp
git commit -m "$(cat <<'EOF'
Add interactive play_cli executable

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: End-to-end integration smoke test

**Files:**
- Create: `tests/integration_smoke.sh`

**Interfaces:**
- Consumes: the `train`, `evaluate`, and `play_cli` binaries (Tasks 8, 9, 7).
- Produces: a shell script that exercises the full pipeline (train → evaluate → play) and fails loudly (non-zero exit) if any stage breaks.

- [ ] **Step 1: Write `tests/integration_smoke.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CHECKPOINT="$(mktemp -t az_integration_XXXXXX.bin)"
trap 'rm -f "$CHECKPOINT"' EXIT

echo "== train (3 tiny iterations) =="
"$BUILD_DIR/train" 3 "$CHECKPOINT"
test -s "$CHECKPOINT"

echo "== evaluate =="
"$BUILD_DIR/evaluate" "$CHECKPOINT" 3

echo "== play_cli (scripted game) =="
echo "0 1 2 3 4 5 6 7 8" | "$BUILD_DIR/play_cli" "$CHECKPOINT"

echo "integration smoke test passed"
```

- [ ] **Step 2: Make it executable and run it**

Run: `chmod +x tests/integration_smoke.sh && ./tests/integration_smoke.sh build`
Expected: prints the `== train ==`, `== evaluate ==`, `== play_cli ==` section headers, each stage's output, and ends with `integration smoke test passed`. Non-zero exit on any stage failure (from `set -e`).

- [ ] **Step 3: Commit**

```bash
git add tests/integration_smoke.sh
git commit -m "$(cat <<'EOF'
Add end-to-end integration smoke test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## After This Plan

Training quality (how close to guaranteed-draw convergence, how many iterations it takes) is a tuning question, not a correctness one — the architecture above is complete and testable end to end. Once implemented, run `./build/train 300 checkpoint.bin` for a real training session and watch the periodic `eval vs minimax` lines converge toward `draws=N losses=0`.
