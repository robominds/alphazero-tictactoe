#include "az/selfplay.hpp"
#include <vector>
#include "az/board.hpp"
#include "az/mcts.hpp"

namespace az {

namespace {
struct PendingExample {
    std::array<float, 18> encoded;
    std::array<float, 9> policy;
    Cell playerToMove;
};
}

void playSelfPlayGame(const Network& network, const SelfPlayConfig& config, ReplayBuffer& buffer) {
    Board board;
    std::vector<PendingExample> pending;

    int ply = 0;
    while (!board.isTerminal()) {
        MCTS mcts(network, config.numSimulations, config.cPuct,
                  /*addRootNoise=*/true, config.dirichletAlpha, config.dirichletEpsilon);
        float temperature = (ply < config.temperatureMoves) ? 1.0f : 0.0f;
        MCTSResult result = mcts.run(board, temperature);
        pending.push_back(PendingExample{board.encode(), result.visitDistribution, board.playerToMove()});
        board = board.applyMove(result.selectedMove);
        ++ply;
    }

    Outcome outcome = board.outcome();
    for (const auto& p : pending) {
        float z;
        if (outcome == Outcome::Draw) {
            z = 0.0f;
        } else {
            bool playerWon = (outcome == Outcome::XWins && p.playerToMove == Cell::X) ||
                              (outcome == Outcome::OWins && p.playerToMove == Cell::O);
            z = playerWon ? 1.0f : -1.0f;
        }
        buffer.add(TrainingExample{p.encoded, p.policy, z});
    }
}

} // namespace az
