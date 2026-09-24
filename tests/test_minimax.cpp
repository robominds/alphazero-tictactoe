#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include "az/board.hpp"
#include "az/minimax.hpp"

using namespace az;

void test_minimax_takes_immediate_win() {
    std::mt19937 rng(1);
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(3); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2
    b = b.applyMove(4); // O
    assert(minimaxBestMove(b, rng) == 2);
}

void test_minimax_blocks_immediate_loss() {
    std::mt19937 rng(1);
    Board b;
    b = b.applyMove(0); // X
    b = b.applyMove(5); // O
    b = b.applyMove(1); // X: X at 0,1, threat at 2 -- O to move must block
    assert(minimaxBestMove(b, rng) == 2);
}

void test_minimax_varies_among_equally_good_moves() {
    // Every opening move draws with perfect play, so all 9 are equally
    // good and repeated calls should eventually pick each of them.
    std::mt19937 rng(12345);
    Board b;
    std::set<int> seen;
    for (int i = 0; i < 200; ++i) seen.insert(minimaxBestMove(b, rng));
    assert(seen.size() == 9);
}

void test_minimax_never_picks_a_worse_move_when_randomizing() {
    // After an X corner opening, the center is O's only drawing reply;
    // every other move loses. Randomizing must never stray from it.
    std::mt19937 rng(12345);
    Board b = Board().applyMove(0);
    for (int i = 0; i < 200; ++i) assert(minimaxBestMove(b, rng) == 4);
}

void test_minimax_never_loses_against_itself() {
    std::mt19937 rng(12345);
    for (int game = 0; game < 20; ++game) {
        Board b;
        while (!b.isTerminal()) {
            b = b.applyMove(minimaxBestMove(b, rng));
        }
        assert(b.outcome() == Outcome::Draw);
    }
}

void test_minimax_rejects_terminal_board() {
    std::mt19937 rng(1);
    Board b;
    int moves[] = {0, 3, 1, 4, 2}; // X completes the top row
    for (int m : moves) b = b.applyMove(m);
    assert(b.isTerminal());

    bool threw = false;
    try {
        minimaxBestMove(b, rng);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_minimax_takes_immediate_win();
    test_minimax_blocks_immediate_loss();
    test_minimax_varies_among_equally_good_moves();
    test_minimax_never_picks_a_worse_move_when_randomizing();
    test_minimax_never_loses_against_itself();
    test_minimax_rejects_terminal_board();
    std::printf("all minimax tests passed\n");
    return 0;
}
