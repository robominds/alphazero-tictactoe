#include <cstdio>
#include <cstdlib>
#include <string>
#include "az/eval.hpp"
#include "az/network.hpp"
#include "az/replay_buffer.hpp"
#include "az/selfplay.hpp"

int main(int argc, char** argv) {
    int numIterations = argc >= 2 ? std::atoi(argv[1]) : 200;
    std::string checkpointPath = argc >= 3 ? argv[2] : "checkpoint.bin";

    const int gamesPerIteration = 25;
    const int batchSize = 32;
    const int trainStepsPerIteration = 20;
    const float learningRate = 0.01f;
    const int evalEveryIterations = 10;
    const int evalGamesPerSide = 20;

    az::Network network;
    az::ReplayBuffer buffer(/*capacity=*/10000);
    az::SelfPlayConfig selfPlayConfig;

    for (int iter = 0; iter < numIterations; ++iter) {
        for (int g = 0; g < gamesPerIteration; ++g) {
            az::playSelfPlayGame(network, selfPlayConfig, buffer);
        }

        if (buffer.size() >= static_cast<size_t>(batchSize)) {
            float lastLoss = 0.0f;
            for (int s = 0; s < trainStepsPerIteration; ++s) {
                auto batch = buffer.sampleBatch(batchSize);
                lastLoss = network.trainStep(batch, learningRate);
            }
            std::printf("iteration %d: buffer=%zu loss=%.4f\n", iter, buffer.size(), lastLoss);
        }

        if ((iter + 1) % evalEveryIterations == 0) {
            az::EvalResult result = az::evaluateAgainstMinimax(network, evalGamesPerSide, /*numSimulations=*/100);
            std::printf("eval vs minimax: wins=%d draws=%d losses=%d\n", result.wins, result.draws, result.losses);
            network.save(checkpointPath);
            std::printf("checkpoint saved to %s\n", checkpointPath.c_str());
        }
    }

    network.save(checkpointPath);
    std::printf("final checkpoint saved to %s\n", checkpointPath.c_str());
    return 0;
}
