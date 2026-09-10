#pragma once
#include <array>
#include <memory>
#include <random>
#include "az/board.hpp"
#include "az/network.hpp"

namespace az {

struct MCTSResult {
    std::array<float, 9> visitDistribution;
    int selectedMove;
};

class MCTS {
public:
    MCTS(const Network& network, int numSimulations, float cPuct = 1.5f);

    // temperature > 0: sample proportional to visitCount^(1/temperature).
    // temperature == 0: pick the max-visit move (ties -> lowest index).
    MCTSResult run(const Board& board, float temperature);

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
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace az
