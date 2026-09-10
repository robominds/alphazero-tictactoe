#include "az/eval.hpp"
#include "az/board.hpp"
#include "az/mcts.hpp"
#include "az/minimax.hpp"

namespace az {

namespace {

int playOneGame(const Network& network, bool networkPlaysX, int numSimulations) {
    Board board;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            MCTS mcts(network, numSimulations, 1.5f);
            move = mcts.run(board, 0.0f).selectedMove;
        } else {
            move = minimaxBestMove(board);
        }
        board = board.applyMove(move);
    }
    Outcome o = board.outcome();
    if (o == Outcome::Draw) return 0;
    bool networkWon = (o == Outcome::XWins && networkPlaysX) || (o == Outcome::OWins && !networkPlaysX);
    return networkWon ? 1 : -1;
}

} // namespace

EvalResult evaluateAgainstMinimax(const Network& network, int gamesPerSide, int numSimulations) {
    EvalResult result;
    auto record = [&](int r) {
        if (r == 1) result.wins++;
        else if (r == -1) result.losses++;
        else result.draws++;
    };
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, true, numSimulations));
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, false, numSimulations));
    return result;
}

} // namespace az
