#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>  // mkdtemp: <stdlib.h> on glibc, <unistd.h> on macOS
#include <vector>
#include "az/network.hpp"

using namespace az;

namespace {

// The two file tests below write into a directory created by mkdtemp(3)
// rather than to a fixed name under the system temp dir. A predictable path
// in a shared, world-writable /tmp lets another local user pre-create it as
// a symlink, and std::ofstream follows symlinks when it opens for writing,
// so the test would truncate whatever the link points at. mkdtemp creates
// the directory atomically with 0700 permissions and fails rather than
// reusing anything already there, which closes both holes at once.
class TempDir {
public:
    TempDir() {
        std::filesystem::path base = std::filesystem::temp_directory_path();
        std::string nameTemplate = (base / "az_test_network_XXXXXX").string();
        if (mkdtemp(nameTemplate.data()) == nullptr) {
            throw std::runtime_error("mkdtemp failed under " + base.string());
        }
        path_ = nameTemplate;
    }

    ~TempDir() {
        // Best effort: a cleanup failure must not mask the test result.
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::filesystem::path file(const char* name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

}  // namespace

void test_predict_output_shapes_and_ranges() {
    Network net;
    std::array<float, 18> input{};
    input[0] = 1.0f;
    Prediction pred = net.predict(input);
    float sum = 0.0f;
    for (float p : pred.policy) {
        assert(p >= 0.0f);
        sum += p;
    }
    assert(std::fabs(sum - 1.0f) < 1e-4f);
    assert(pred.value >= -1.0f && pred.value <= 1.0f);
}

void test_train_step_reduces_loss_on_fixed_batch() {
    Network net;
    TrainingExample example;
    example.encodedBoard.fill(0.0f);
    example.encodedBoard[0] = 1.0f;
    example.targetPolicy.fill(0.0f);
    example.targetPolicy[4] = 1.0f;
    example.targetValue = 1.0f;

    std::vector<TrainingExample> batch{example};

    float firstLoss = net.trainStep(batch, 0.05f);
    float lastLoss = firstLoss;
    for (int i = 0; i < 200; ++i) {
        lastLoss = net.trainStep(batch, 0.05f);
    }
    assert(lastLoss < firstLoss);
    assert(lastLoss < 0.1f);
}

void test_save_load_round_trip_is_bit_exact(const TempDir& tmp) {
    Network net;
    std::array<float, 18> input{};
    input[0] = 1.0f;
    input[5] = -1.0f;
    Prediction before = net.predict(input);

    std::filesystem::path path = tmp.file("roundtrip.bin");
    net.save(path.string());

    Network loaded;
    loaded.load(path.string());
    Prediction after = loaded.predict(input);

    for (int k = 0; k < 9; ++k) {
        assert(before.policy[k] == after.policy[k]);
    }
    assert(before.value == after.value);
}

void test_load_rejects_corrupt_or_wrong_format_file(const TempDir& tmp) {
    std::filesystem::path path = tmp.file("garbage.bin");
    {
        std::ofstream out(path, std::ios::binary);
        // Wrong magic, but plausible length so a naive size check wouldn't
        // catch it -- pad with junk bytes.
        const char garbage[] = "NOTANETWORKCHECKPOINTFILE-------------------------------";
        out.write(garbage, sizeof(garbage));
    }

    Network net;
    bool threw = false;
    try {
        net.load(path.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    TempDir tmp;
    test_predict_output_shapes_and_ranges();
    test_train_step_reduces_loss_on_fixed_batch();
    test_save_load_round_trip_is_bit_exact(tmp);
    test_load_rejects_corrupt_or_wrong_format_file(tmp);
    std::printf("all network tests passed\n");
    return 0;
}
