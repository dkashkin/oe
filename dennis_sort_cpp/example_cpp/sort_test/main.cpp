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
    std::uniform_int_distribution<int> dist(-10000, 10000);
    std::vector<int> data(size);
    for (auto& v : data) v = dist(rng);
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
    std::uniform_int_distribution<int> dist(0, unique_values - 1);
    std::vector<int> data(size);
    for (auto& v : data) v = dist(rng);
    return data;
}

std::vector<int> generate_partially_sorted_data(size_t size, double sorted_fraction) {
    std::mt19937 rng(std::random_device{}());
    size_t sorted_size = static_cast<size_t>(size * sorted_fraction);
    std::vector<int> data;
    data.reserve(size);

    for (size_t i = 0; i < sorted_size; ++i) data.push_back(static_cast<int>(i));

    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (size_t i = sorted_size; i < size; ++i) data.push_back(dist(rng));

    return data;
}

int main() {
    std::vector<std::vector<int>> test_data = {
        generate_random_data(10000),
        generate_random_data(100000),
        generate_nearly_sorted_data(10000, 0.05),
        generate_nearly_sorted_data(100000, 0.05),
        generate_reverse_sorted_data(10000),
        generate_reverse_sorted_data(100000),
        generate_data_with_duplicates(10000, 10),
        generate_data_with_duplicates(100000, 100),
        generate_partially_sorted_data(10000, 0.3),
        generate_partially_sorted_data(100000, 0.3),
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
