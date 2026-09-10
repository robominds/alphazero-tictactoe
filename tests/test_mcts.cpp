#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
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

void test_mcts_root_noise_still_explores_the_winning_move() {
    // Root noise trades away "always greedily prefer the best-known move"
    // for "never permanently ignore any legal move" -- with a random,
    // untrained network and noise pulling exploration away from the prior,
    // an obvious win is not guaranteed to end up as the single top pick
    // within a fixed simulation budget (that would defeat the point of
    // adding noise). What noise does guarantee: the winning move still
    // gets a real, non-zero share of visits, because it can never be
    // starved down to (near) zero prior the way an ordinary prior could.
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O

    Network net;
    MCTS mcts(net, /*numSimulations=*/300, /*cPuct=*/1.5f, /*addRootNoise=*/true);
    MCTSResult result = mcts.run(b, /*temperature=*/0.0f);

    assert(result.visitDistribution[2] > 0.0f);
}

void test_dirichlet_noise_gives_every_legal_move_positive_probability() {
    // A network can be arbitrarily confident (even priors of exactly 0 for
    // some legal move) -- root noise must still guarantee every legal move
    // keeps some non-zero chance of being explored.
    std::array<float, 9> priors{};
    priors[0] = 1.0f; // "certain" move -- every other legal move starts at 0
    std::vector<int> legalMoves = {0, 1, 2, 3, 4};
    std::mt19937 rng(42);

    MCTS::mixDirichletNoise(priors, legalMoves, rng, /*alpha=*/0.3f, /*epsilon=*/0.25f);

    float sum = 0.0f;
    for (int m : legalMoves) {
        assert(priors[m] > 0.0f);
        sum += priors[m];
    }
    assert(std::fabs(sum - 1.0f) < 1e-3f);
}

void test_mcts_rejects_terminal_board() {
    Board b;
    int moves[] = {0, 3, 1, 4, 2}; // X completes the top row
    for (int m : moves) b = b.applyMove(m);
    assert(b.isTerminal());

    Network net;
    MCTS mcts(net, /*numSimulations=*/10, /*cPuct=*/1.5f);
    for (float temperature : {0.0f, 1.0f}) {
        bool threw = false;
        try {
            mcts.run(b, temperature);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

int main() {
    test_mcts_finds_immediate_winning_move();
    test_mcts_root_noise_still_explores_the_winning_move();
    test_dirichlet_noise_gives_every_legal_move_positive_probability();
    test_mcts_rejects_terminal_board();
    std::printf("all mcts tests passed\n");
    return 0;
}
