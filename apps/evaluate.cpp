#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include "az/eval.hpp"
#include "az/network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: evaluate <checkpoint-path> [games-per-side]\n");
        return 1;
    }
    std::string checkpointPath = argv[1];
    int gamesPerSide = argc >= 3 ? std::atoi(argv[2]) : 50;

    az::Network network;
    try {
        network.load(checkpointPath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    az::EvalResult result = az::evaluateAgainstMinimax(network, gamesPerSide, /*numSimulations=*/100);
    std::printf("vs minimax over %d games/side: wins=%d draws=%d losses=%d\n",
                gamesPerSide, result.wins, result.draws, result.losses);
    return 0;
}
