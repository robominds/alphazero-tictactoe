#pragma once
#include "az/board.hpp"

namespace az {

// Returns the game-theoretically optimal move for board.playerToMove(),
// via exhaustive minimax search. Precondition: !board.isTerminal().
int minimaxBestMove(const Board& board);

} // namespace az
