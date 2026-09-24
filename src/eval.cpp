#include "az/eval.hpp"
#include <random>
#include "az/board.hpp"
#include "az/mcts.hpp"
#include "az/minimax.hpp"

namespace az {

namespace {

int playOneGame(const Network& network, bool networkPlaysX, int numSimulations, std::mt19937& rng) {
    Board board;
    while (!board.isTerminal()) {
        bool networkTurn = (board.playerToMove() == Cell::X) == networkPlaysX;
        int move;
        if (networkTurn) {
            MCTS mcts(network, numSimulations, 1.5f);
            move = mcts.run(board, 0.0f).selectedMove;
        } else {
            move = minimaxBestMove(board, rng);
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
    // Minimax breaks ties between equally good moves at random, so each
    // game can follow a different line; with fixed tie-breaking every game
    // per side would be the same game, since greedy search is deterministic.
    std::mt19937 rng{std::random_device{}()};
    auto record = [&](int r) {
        if (r == 1) result.wins++;
        else if (r == -1) result.losses++;
        else result.draws++;
    };
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, true, numSimulations, rng));
    for (int i = 0; i < gamesPerSide; ++i) record(playOneGame(network, false, numSimulations, rng));
    return result;
}

} // namespace az
