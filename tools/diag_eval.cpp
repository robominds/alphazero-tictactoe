#include <cstdio>
#include <string>
#include "az/board.hpp"
#include "az/network.hpp"
#include "az/mcts.hpp"
#include "az/minimax.hpp"

namespace {

void playAndPrint(const az::Network& net, bool networkPlaysX) {
    az::Board board;
    std::printf("=== network plays %s ===\n", networkPlaysX ? "X" : "O");
    int ply = 0;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == az::Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            az::MCTS mcts(net, 100, 1.5f);
            auto result = mcts.run(board, 0.0f);
            move = result.selectedMove;
            std::printf("ply %d: network plays %d  (visits:", ply, move);
            for (int m : board.legalMoves()) {
                std::printf(" %d=%.2f", m, result.visitDistribution[m]);
            }
            std::printf(")\n");
        } else {
            move = az::minimaxBestMove(board);
            std::printf("ply %d: minimax plays %d\n", ply, move);
        }
        board = board.applyMove(move);
        ++ply;
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            az::Cell cell = board.cellAt(r * 3 + c);
            char ch = cell == az::Cell::X ? 'X' : cell == az::Cell::O ? 'O' : '.';
            std::printf("%c ", ch);
        }
        std::printf("\n");
    }
    az::Outcome o = board.outcome();
    std::printf("outcome: %s\n\n",
                 o == az::Outcome::Draw ? "Draw" : (o == az::Outcome::XWins ? "X wins" : "O wins"));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: diag_eval <checkpoint-path>\n");
        return 1;
    }
    az::Network net;
    try {
        net.load(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    playAndPrint(net, true);
    playAndPrint(net, false);
    return 0;
}
