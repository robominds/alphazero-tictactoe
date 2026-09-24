#include "az/minimax.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

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

int minimaxBestMove(const Board& board, std::mt19937& rng) {
    // A terminal board has no legal moves, so bestMoves below would stay
    // empty and the pick would index past it. Throw rather than assert: this file compiles
    // with NDEBUG in Release, where an assert is a no-op.
    std::vector<int> moves = board.legalMoves();
    if (moves.empty()) {
        throw std::invalid_argument("minimaxBestMove: board is terminal, no move to pick");
    }
    int bestScore = std::numeric_limits<int>::min();
    std::vector<int> bestMoves;
    for (int m : moves) {
        int score = -scoreOf(board.applyMove(m));
        if (score > bestScore) {
            bestScore = score;
            bestMoves.clear();
        }
        if (score == bestScore) bestMoves.push_back(m);
    }
    std::uniform_int_distribution<std::size_t> pick(0, bestMoves.size() - 1);
    return bestMoves[pick(rng)];
}

} // namespace az
