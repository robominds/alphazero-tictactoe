#include "az/mcts.hpp"
#include <algorithm>
#include <cmath>

namespace az {

MCTS::MCTS(const Network& network, int numSimulations, float cPuct,
           bool addRootNoise, float dirichletAlpha, float dirichletEpsilon)
    : network_(network), numSimulations_(numSimulations), cPuct_(cPuct),
      addRootNoise_(addRootNoise), dirichletAlpha_(dirichletAlpha), dirichletEpsilon_(dirichletEpsilon) {}

void MCTS::mixDirichletNoise(std::array<float, 9>& priors, const std::vector<int>& legalMoves,
                              std::mt19937& rng, float alpha, float epsilon) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::array<float, 9> noise{};
    float sum = 0.0f;
    for (int m : legalMoves) {
        // A Gamma(alpha,1) draw is 0 with probability 0, but numerically it
        // can land extremely close to it; floor it so every legal move is
        // guaranteed strictly positive noise mass.
        noise[m] = std::max(gamma(rng), 1e-6f);
        sum += noise[m];
    }
    for (int m : legalMoves) {
        float noiseFrac = noise[m] / sum;
        priors[m] = (1.0f - epsilon) * priors[m] + epsilon * noiseFrac;
    }
}

float MCTS::expand(Node& node) {
    Prediction pred = network_.predict(node.board.encode());
    std::array<float, 9> masked{};
    float sum = 0.0f;
    for (int m : node.board.legalMoves()) {
        masked[m] = pred.policy[m];
        sum += masked[m];
    }
    if (sum > 1e-8f) {
        for (int m : node.board.legalMoves()) masked[m] /= sum;
    } else {
        float uniform = 1.0f / static_cast<float>(node.board.legalMoves().size());
        for (int m : node.board.legalMoves()) masked[m] = uniform;
    }
    node.priors = masked;
    node.expanded = true;
    return pred.value;
}

int MCTS::selectChild(const Node& node) const {
    int totalVisits = 0;
    for (int m : node.board.legalMoves()) totalVisits += node.visitCounts[m];

    float bestScore = -1e9f;
    int bestMove = node.board.legalMoves().front();
    for (int m : node.board.legalMoves()) {
        float q = node.visitCounts[m] > 0 ? node.totalValue[m] / node.visitCounts[m] : 0.0f;
        float u = cPuct_ * node.priors[m] * std::sqrt(static_cast<float>(totalVisits) + 1e-8f)
                  / (1 + node.visitCounts[m]);
        float score = q + u;
        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
        }
    }
    return bestMove;
}

float MCTS::simulate(Node& node) {
    if (node.board.isTerminal()) {
        return node.board.outcome() == Outcome::Draw ? 0.0f : -1.0f;
    }
    if (!node.expanded) {
        return expand(node);
    }
    int move = selectChild(node);
    if (!node.children[move]) {
        node.children[move] = std::make_unique<Node>();
        node.children[move]->board = node.board.applyMove(move);
    }
    float childValue = simulate(*node.children[move]);
    float value = -childValue;
    node.visitCounts[move] += 1;
    node.totalValue[move] += value;
    return value;
}

MCTSResult MCTS::run(const Board& board, float temperature) {
    Node root;
    root.board = board;
    expand(root); // pre-expand so root noise (if any) applies before any simulation runs
    if (addRootNoise_) {
        mixDirichletNoise(root.priors, board.legalMoves(), rng_, dirichletAlpha_, dirichletEpsilon_);
    }
    for (int i = 0; i < numSimulations_; ++i) {
        simulate(root);
    }

    std::array<float, 9> dist{};
    int totalVisits = 0;
    for (int m : board.legalMoves()) totalVisits += root.visitCounts[m];
    if (totalVisits > 0) {
        for (int m : board.legalMoves()) {
            dist[m] = static_cast<float>(root.visitCounts[m]) / static_cast<float>(totalVisits);
        }
    } else {
        float uniform = 1.0f / static_cast<float>(board.legalMoves().size());
        for (int m : board.legalMoves()) dist[m] = uniform;
    }

    int selected;
    if (temperature <= 0.0f) {
        selected = board.legalMoves().front();
        int bestVisits = -1;
        for (int m : board.legalMoves()) {
            if (root.visitCounts[m] > bestVisits) {
                bestVisits = root.visitCounts[m];
                selected = m;
            }
        }
    } else {
        std::array<float, 9> weights{};
        float sum = 0.0f;
        for (int m : board.legalMoves()) {
            weights[m] = std::pow(static_cast<float>(root.visitCounts[m]), 1.0f / temperature);
            sum += weights[m];
        }
        std::uniform_real_distribution<float> unif(0.0f, sum);
        float r = unif(rng_);
        float cumulative = 0.0f;
        selected = board.legalMoves().back();
        for (int m : board.legalMoves()) {
            cumulative += weights[m];
            if (r <= cumulative) {
                selected = m;
                break;
            }
        }
    }

    return MCTSResult{dist, selected};
}

} // namespace az
