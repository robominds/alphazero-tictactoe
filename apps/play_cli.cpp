#include <cstdio>
#include <cstdlib>
#include <exception>
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
    try {
        network.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

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
                    if (std::cin.eof()) {
                        std::printf("\nInput ended, exiting.\n");
                        return 0;
                    }
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
