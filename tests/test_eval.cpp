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
