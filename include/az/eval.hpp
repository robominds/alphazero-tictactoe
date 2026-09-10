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
