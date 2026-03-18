#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <vector>

// Copy the generator functions (without the sort-related includes)
std::vector<int> generate_random_data(size_t size) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
    std::vector<int> data(size);
    for (auto& v : data) v = dist(rng) % 10000;
    return data;
}

std::vector<int> generate_nearly_sorted_data(size_t size, double disorder_rate) {
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<int>(i);
    size_t swaps = static_cast<size_t>(size * disorder_rate);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, size - 1);
    for (size_t s = 0; s < swaps; ++s) {
        std::swap(data[dist(rng)], data[dist(rng)]);
    }
    return data;
}

std::vector<int> generate_reverse_sorted_data(size_t size) {
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<int>(size - i - 1);
    return data;
}

std::vector<int> generate_data_with_duplicates(size_t size, int unique_values) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
    std::vector<int> data(size);
    for (auto& v : data) v = dist(rng) % unique_values;
    return data;
}

std::vector<int> generate_partially_sorted_data(size_t size, double sorted_fraction) {
    std::mt19937 rng(std::random_device{}());
    size_t sorted_size = static_cast<size_t>(size * sorted_fraction);
    std::vector<int> data;
    data.reserve(size);
    for (size_t i = 0; i < sorted_size; ++i) data.push_back(static_cast<int>(i));
    std::uniform_int_distribution<int32_t> dist(INT32_MIN, INT32_MAX);
    for (size_t i = sorted_size; i < size; ++i) data.push_back(dist(rng) % 10000);
    return data;
}

std::vector<int> generate_organ_pipe_data(size_t size) {
    std::vector<int> data(size);
    size_t mid = size / 2;
    for (size_t i = 0; i < mid; ++i) data[i] = static_cast<int>(i);
    for (size_t i = mid; i < size; ++i) data[i] = static_cast<int>(size - i - 1);
    return data;
}

std::vector<int> generate_interleaved_sorted_data(size_t size, int k) {
    std::vector<int> data;
    data.reserve(size);
    size_t chunk = size / k;
    std::vector<std::vector<int>> runs(k);
    for (int i = 0; i < k; ++i) {
        for (size_t j = 0; j < chunk; ++j) {
            runs[i].push_back(static_cast<int>(j * k + i));
        }
    }
    for (size_t j = 0; j < chunk; ++j) {
        for (int i = 0; i < k; ++i) data.push_back(runs[i][j]);
    }
    return data;
}

std::vector<int> generate_shuffled_blocks_data(size_t size, size_t block_size) {
    std::vector<int> data(size);
    for (size_t i = 0; i < size; ++i) data[i] = static_cast<int>(i);
    size_t num_blocks = size / block_size;
    std::vector<size_t> block_indices(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) block_indices[i] = i;
    std::mt19937 rng(std::random_device{}());
    std::shuffle(block_indices.begin(), block_indices.end(), rng);
    std::vector<int> result;
    result.reserve(size);
    for (size_t bi : block_indices) {
        for (size_t j = 0; j < block_size; ++j)
            result.push_back(data[bi * block_size + j]);
    }
    return result;
}

std::vector<int> generate_skewed_data(size_t size, double exponent) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<int> data(size);
    for (auto& v : data) {
        double u = dist(rng);
        v = static_cast<int>(std::pow(u, exponent) * 1000000);
    }
    return data;
}

std::vector<int> generate_sawtooth_data(size_t size, size_t tooth_size) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> offset_dist(0, 100000);
    std::vector<int> data;
    data.reserve(size);
    for (size_t i = 0; i < size; ) {
        int base = offset_dist(rng);
        for (size_t j = 0; j < tooth_size && i < size; ++j, ++i) {
            data.push_back(base + static_cast<int>(j));
        }
    }
    return data;
}

// Test helpers
int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const char* test_name) {
    if (condition) {
        std::cout << "  PASS: " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: " << test_name << std::endl;
        tests_failed++;
    }
}

int main() {
    const size_t N = 1000;

    // --- Test generate_random_data ---
    std::cout << "[generate_random_data]" << std::endl;
    {
        auto data = generate_random_data(N);
        check(data.size() == N, "correct size");
        check(!std::is_sorted(data.begin(), data.end()), "not sorted");
        bool in_range = true;
        for (int v : data) { if (v < -9999 || v > 9999) { in_range = false; break; } }
        check(in_range, "values in [-9999, 9999]");
    }

    // --- Test generate_nearly_sorted_data ---
    std::cout << "[generate_nearly_sorted_data]" << std::endl;
    {
        auto data = generate_nearly_sorted_data(N, 0.05);
        check(data.size() == N, "correct size");
        // Most elements should be near their sorted position
        int displaced = 0;
        for (size_t i = 0; i < N; ++i) {
            if (data[i] != static_cast<int>(i)) displaced++;
        }
        double displaced_frac = static_cast<double>(displaced) / N;
        check(displaced_frac < 0.20, "less than 20% displaced (5% swap rate)");
        check(displaced_frac > 0.0, "at least some elements displaced");
        // Should contain all values 0..N-1 (permutation)
        std::set<int> vals(data.begin(), data.end());
        check(vals.size() == N, "is a permutation of 0..N-1");
    }

    // --- Test generate_reverse_sorted_data ---
    std::cout << "[generate_reverse_sorted_data]" << std::endl;
    {
        auto data = generate_reverse_sorted_data(N);
        check(data.size() == N, "correct size");
        check(data[0] == static_cast<int>(N - 1), "first element is N-1");
        check(data[N - 1] == 0, "last element is 0");
        bool strictly_desc = true;
        for (size_t i = 1; i < N; ++i) {
            if (data[i] >= data[i - 1]) { strictly_desc = false; break; }
        }
        check(strictly_desc, "strictly descending");
    }

    // --- Test generate_data_with_duplicates ---
    std::cout << "[generate_data_with_duplicates]" << std::endl;
    {
        auto data = generate_data_with_duplicates(N, 10);
        check(data.size() == N, "correct size");
        std::set<int> unique_vals(data.begin(), data.end());
        // % 10 on signed ints yields values in [-9, 9], so up to 19 unique values
        check(unique_vals.size() <= 19, "at most 2*unique_values-1 unique values");
        check(unique_vals.size() > 1, "more than 1 unique value");
    }

    // --- Test generate_partially_sorted_data ---
    std::cout << "[generate_partially_sorted_data]" << std::endl;
    {
        auto data = generate_partially_sorted_data(N, 0.3);
        check(data.size() == N, "correct size");
        size_t sorted_size = static_cast<size_t>(N * 0.3);
        bool prefix_sorted = true;
        for (size_t i = 0; i < sorted_size; ++i) {
            if (data[i] != static_cast<int>(i)) { prefix_sorted = false; break; }
        }
        check(prefix_sorted, "first 30% is sorted sequence 0..k");
    }

    // --- Test generate_organ_pipe_data ---
    std::cout << "[generate_organ_pipe_data]" << std::endl;
    {
        auto data = generate_organ_pipe_data(N);
        check(data.size() == N, "correct size");
        size_t mid = N / 2;
        bool first_half_asc = true;
        for (size_t i = 1; i < mid; ++i) {
            if (data[i] <= data[i - 1]) { first_half_asc = false; break; }
        }
        check(first_half_asc, "first half is ascending");
        bool second_half_desc = true;
        for (size_t i = mid + 1; i < N; ++i) {
            if (data[i] >= data[i - 1]) { second_half_desc = false; break; }
        }
        check(second_half_desc, "second half is descending");
        check(data[mid - 1] >= data[mid], "peak at midpoint");
    }

    // --- Test generate_interleaved_sorted_data ---
    std::cout << "[generate_interleaved_sorted_data]" << std::endl;
    {
        int k = 4;
        size_t sz = 100; // use small size for easy verification
        auto data = generate_interleaved_sorted_data(sz, k);
        size_t chunk = sz / k;
        check(data.size() == chunk * k, "correct size (chunk*k)");
        // Each stride-k subsequence should be sorted
        bool strides_sorted = true;
        for (int s = 0; s < k; ++s) {
            for (size_t j = 1; j < chunk; ++j) {
                if (data[j * k + s] <= data[(j - 1) * k + s]) {
                    strides_sorted = false;
                    break;
                }
            }
        }
        check(strides_sorted, "each stride-k subsequence is sorted");
        // Should contain all values 0..sz-1
        std::set<int> vals(data.begin(), data.end());
        check(vals.size() == chunk * k, "all values are unique");
    }

    // --- Test generate_shuffled_blocks_data ---
    std::cout << "[generate_shuffled_blocks_data]" << std::endl;
    {
        size_t block_size = 100;
        auto data = generate_shuffled_blocks_data(N, block_size);
        check(data.size() == N, "correct size");
        // Each block of block_size elements should be internally sorted
        size_t num_blocks = N / block_size;
        bool blocks_sorted = true;
        for (size_t b = 0; b < num_blocks; ++b) {
            for (size_t j = 1; j < block_size; ++j) {
                if (data[b * block_size + j] <= data[b * block_size + j - 1]) {
                    blocks_sorted = false;
                    break;
                }
            }
            if (!blocks_sorted) break;
        }
        check(blocks_sorted, "each block is internally sorted");
        // Overall should NOT be sorted (blocks are shuffled)
        check(!std::is_sorted(data.begin(), data.end()), "not globally sorted");
        // Should be a permutation of 0..N-1
        std::set<int> vals(data.begin(), data.end());
        check(vals.size() == N, "is a permutation of 0..N-1");
    }

    // --- Test generate_skewed_data ---
    std::cout << "[generate_skewed_data]" << std::endl;
    {
        auto data = generate_skewed_data(N, 3.0);
        check(data.size() == N, "correct size");
        bool in_range = true;
        for (int v : data) { if (v < 0 || v > 1000000) { in_range = false; break; } }
        check(in_range, "values in [0, 1000000]");
        // With exponent=3, most values should be small (near 0)
        // With exponent=3, P(V <= 500000) = 0.5^(1/3) ~ 0.794, so >70% should be below midpoint
        int below_threshold = 0;
        for (int v : data) { if (v < 500000) below_threshold++; }
        double frac_below = static_cast<double>(below_threshold) / N;
        check(frac_below > 0.70, "distribution is skewed toward zero");
    }

    // --- Test generate_sawtooth_data ---
    std::cout << "[generate_sawtooth_data]" << std::endl;
    {
        size_t tooth = 50;
        auto data = generate_sawtooth_data(N, tooth);
        check(data.size() == N, "correct size");
        // Each tooth should be a short ascending run
        size_t num_teeth = N / tooth;
        bool teeth_ascending = true;
        for (size_t t = 0; t < num_teeth; ++t) {
            for (size_t j = 1; j < tooth; ++j) {
                if (data[t * tooth + j] != data[t * tooth + j - 1] + 1) {
                    teeth_ascending = false;
                    break;
                }
            }
            if (!teeth_ascending) break;
        }
        check(teeth_ascending, "each tooth is a consecutive ascending run");
        // Should NOT be globally sorted (random offsets)
        check(!std::is_sorted(data.begin(), data.end()), "not globally sorted");
    }

    // --- Summary ---
    std::cout << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    return tests_failed > 0 ? 1 : 0;
}
