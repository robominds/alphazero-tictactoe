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
