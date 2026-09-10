#include <cassert>
#include <cstdio>
#include <stdexcept>
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

void test_minimax_rejects_terminal_board() {
    Board b;
    int moves[] = {0, 3, 1, 4, 2}; // X completes the top row
    for (int m : moves) b = b.applyMove(m);
    assert(b.isTerminal());

    bool threw = false;
    try {
        minimaxBestMove(b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_minimax_takes_immediate_win();
    test_minimax_blocks_immediate_loss();
    test_minimax_never_loses_against_itself();
    test_minimax_rejects_terminal_board();
    std::printf("all minimax tests passed\n");
    return 0;
}
