#pragma once
#include <random>
#include "az/board.hpp"

namespace az {

// Returns a game-theoretically optimal move for board.playerToMove(),
// via exhaustive minimax search. When several moves are equally good,
// picks one uniformly at random using rng. Precondition:
// !board.isTerminal(); throws std::invalid_argument if the board is
// terminal.
int minimaxBestMove(const Board& board, std::mt19937& rng);

} // namespace az
