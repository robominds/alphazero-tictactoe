#include <cstdio>
#include <exception>
#include <string>
#include "arg_parse.hpp"
#include "az/eval.hpp"
#include "az/network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: evaluate <checkpoint-path> [games-per-side]\n");
        return 1;
    }
    std::string checkpointPath = argv[1];
    int gamesPerSide = 50;
    if (argc >= 3) {
        std::optional<int> parsed = az::parsePositiveIntArg(argv[2]);
        if (!parsed) {
            std::fprintf(stderr, "usage: evaluate <checkpoint-path> [games-per-side]\n");
            std::fprintf(stderr, "error: games-per-side must be a positive integer, got \"%s\"\n", argv[2]);
            return 1;
        }
        gamesPerSide = *parsed;
    }

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
