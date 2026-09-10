#pragma once
#include <array>
#include <memory>
#include <random>
#include <vector>
#include "az/board.hpp"
#include "az/network.hpp"

namespace az {

struct MCTSResult {
    std::array<float, 9> visitDistribution;
    int selectedMove;
};

class MCTS {
public:
    // addRootNoise mixes Dirichlet(alpha) noise into the root's priors
    // before search begins (epsilon-weighted: (1-epsilon)*prior +
    // epsilon*noise), matching AlphaZero's self-play exploration. Without
    // it, a legal move the network assigns a near-zero prior can end up
    // with a near-zero PUCT exploration bonus too, so search may never
    // visit it enough to discover it's actually necessary -- exactly the
    // failure mode this guards against.
    MCTS(const Network& network, int numSimulations, float cPuct = 1.5f,
         bool addRootNoise = false, float dirichletAlpha = 0.3f, float dirichletEpsilon = 0.25f);

    // temperature > 0: sample proportional to visitCount^(1/temperature).
    // temperature == 0: pick the max-visit move (ties -> lowest index).
    // Precondition: !board.isTerminal(); throws std::invalid_argument if
    // the board is terminal, since there is then no move to return.
    MCTSResult run(const Board& board, float temperature);

    // Exposed for testing: mixes Dirichlet(alpha) noise into priors over
    // legalMoves in place, weighted by epsilon. Guarantees every entry in
    // legalMoves ends up with strictly positive probability, regardless of
    // its starting value.
    static void mixDirichletNoise(std::array<float, 9>& priors, const std::vector<int>& legalMoves,
                                   std::mt19937& rng, float alpha, float epsilon);

private:
    struct Node {
        Board board;
        bool expanded = false;
        std::array<float, 9> priors{};
        std::array<int, 9> visitCounts{};
        std::array<float, 9> totalValue{};
        std::array<std::unique_ptr<Node>, 9> children{};
    };

    float expand(Node& node);
    float simulate(Node& node);
    int selectChild(const Node& node) const;

    const Network& network_;
    int numSimulations_;
    float cPuct_;
    bool addRootNoise_;
    float dirichletAlpha_;
    float dirichletEpsilon_;
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace az
