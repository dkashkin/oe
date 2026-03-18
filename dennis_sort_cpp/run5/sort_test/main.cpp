#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

// Forward declarations for functions defined in the evolving sort implementation
void adaptive_sort(std::vector<int>& arr);
void quicksort(std::vector<int>& arr, int low, int high);
int partition(std::vector<int>& arr, int low, int high);
bool is_nearly_sorted(const std::vector<int>& arr, double threshold);
void insertion_sort(std::vector<int>& arr);

// Include the evolving sort implementation
#include "sort_impl.cpp"

// Benchmark infrastructure

struct BenchmarkResults {
    std::vector<double> times;
    std::vector<bool> correctness;
    double adaptability_score = 0.0;
};

BenchmarkResults run_benchmark(const std::vector<std::vector<int>>& test_data) {
    BenchmarkResults results;

    for (const auto& data : test_data) {
        std::vector<int> arr = data;
        auto start = std::chrono::high_resolution_clock::now();

        adaptive_sort(arr);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        results.times.push_back(elapsed);

        // Check if correctly sorted
        bool is_sorted = std::is_sorted(arr.begin(), arr.end());
        results.correctness.push_back(is_sorted);
    }

    // Calculate adaptability score based on performance variance
    if (results.times.size() > 1) {
        double sum = 0.0;
        for (double t : results.times) sum += t;
        double mean_time = sum / results.times.size();

        double variance = 0.0;
        for (double t : results.times) {
            double diff = t - mean_time;
            variance += diff * diff;
        }
        variance /= results.times.size();

        results.adaptability_score = 1.0 / (1.0 + std::sqrt(variance));
    }

    return results;
}

// Data generators

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

int main() {
    std::vector<std::vector<int>> test_data = {
        generate_random_data(20000),
        generate_random_data(200000),
        generate_nearly_sorted_data(20000, 0.05),
        generate_nearly_sorted_data(200000, 0.05),
        generate_reverse_sorted_data(20000),
        generate_reverse_sorted_data(200000),
        generate_data_with_duplicates(20000, 10),
        generate_data_with_duplicates(200000, 100),
        generate_partially_sorted_data(20000, 0.3),
        generate_partially_sorted_data(200000, 0.3),
        generate_organ_pipe_data(20000),
        generate_organ_pipe_data(200000),
        generate_interleaved_sorted_data(20000, 8),
        generate_interleaved_sorted_data(200000, 8),
        generate_shuffled_blocks_data(20000, 1000),
        generate_shuffled_blocks_data(200000, 1000),
        generate_skewed_data(20000, 3.0),
        generate_skewed_data(200000, 3.0),
        generate_sawtooth_data(20000, 50),
        generate_sawtooth_data(200000, 50),
    };

    BenchmarkResults results = run_benchmark(test_data);

    bool all_correct = true;
    for (bool c : results.correctness) {
        if (!c) { all_correct = false; break; }
    }
    double correctness_score = all_correct ? 1.0 : 0.0;

    double sum = 0.0;
    for (double t : results.times) sum += t;
    double avg_time = sum / results.times.size();

    double performance_score = 1.0 / (1.0 + avg_time * 10.0);

    // Output results as JSON
    std::cout << "{" << std::endl;
    std::cout << "  \"correctness\": " << correctness_score << "," << std::endl;
    std::cout << "  \"avg_time\": " << avg_time << "," << std::endl;
    std::cout << "  \"performance_score\": " << performance_score << "," << std::endl;
    std::cout << "  \"adaptability_score\": " << results.adaptability_score << "," << std::endl;
    std::cout << "  \"times\": [";
    for (size_t i = 0; i < results.times.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << results.times[i];
    }
    std::cout << "]," << std::endl;
    std::cout << "  \"all_correct\": " << (all_correct ? "true" : "false") << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}
