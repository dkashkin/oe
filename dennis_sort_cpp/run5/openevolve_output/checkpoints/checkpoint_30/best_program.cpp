#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    const size_t n = arr.size();
    if (n < 2) return;

    // Fast O(N) detection of pre-sorted or reverse-sorted patterns
    // This reliably breaks in O(1) for random data
    bool asc = true, desc = true;
    for (size_t i = 1; i < n; ++i) {
        if (arr[i-1] > arr[i]) asc = false;
        if (arr[i-1] < arr[i]) desc = false;
        if (!(asc || desc)) break;
    }
    if (asc) return;
    if (desc) {
        std::reverse(arr.begin(), arr.end());
        return;
    }

    // Density check: calculate range to determine if Counting Sort is viable
    int min_v = arr[0], max_v = arr[0];
    for (size_t i = 1; i < n; ++i) {
        min_v = std::min(min_v, arr[i]);
        max_v = std::max(max_v, arr[i]);
    }

    const long long range = (long long)max_v - min_v;
    // Use Counting Sort for dense distributions (memory-safe threshold of ~8MB)
    if (range < 2000000 && range < (long long)n * 20) {
        std::vector<int> counts(range + 1, 0);
        for (int x : arr) {
            counts[x - min_v]++;
        }
        auto it = arr.begin();
        for (int i = 0; i <= (int)range; ++i) {
            if (int c = counts[i]) {
                std::fill_n(it, c, i + min_v);
                it += c;
            }
        }
        return;
    }

    // Fallback to Introsort for sparse distributions
    std::sort(arr.begin(), arr.end());
}