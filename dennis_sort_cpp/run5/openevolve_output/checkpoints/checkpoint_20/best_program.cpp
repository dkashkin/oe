#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    int n = arr.size();
    if (n < 2) return;

    // Fast O(N) check for sorted or strictly reverse-sorted inputs
    // On random data, this reliably breaks in ~2.5 iterations (O(1) overhead)
    bool asc = true;
    bool desc = true;
    for (int i = 1; i < n; ++i) {
        if (arr[i-1] > arr[i]) asc = false;
        if (arr[i-1] < arr[i]) desc = false;
        if (!asc && !desc) break;
    }
    
    if (asc) return;
    if (desc) {
        std::reverse(arr.begin(), arr.end());
        return;
    }

    // Attempt Counting Sort for inputs with a dense distribution
    if (n >= 128) {
        int min_val = arr[0];
        int max_val = arr[0];
        // std::min/max vectorize extremely well to vminps/vmaxps under AVX2
        for (int i = 1; i < n; ++i) {
            min_val = std::min(min_val, arr[i]);
            max_val = std::max(max_val, arr[i]);
        }

        long long range = (long long)max_val - (long long)min_val;
        
        // If density is sufficient, solve O(N log N) into strict branchless O(N)
        // Hard-cap memory allocation overhead to 8MB max (2 million int array)
        if (range <= 2000000 && range <= (long long)n * 10) {
            std::vector<int> counts(range + 1, 0);
            for (int i = 0; i < n; ++i) {
                counts[(long long)arr[i] - min_val]++;
            }
            int idx = 0;
            for (int i = 0; i <= range; ++i) {
                int c = counts[i];
                if (c > 0) {
                    std::fill(arr.begin() + idx, arr.begin() + idx + c, i + min_val);
                    idx += c;
                }
            }
            return;
        }
    }

    // Fallback highly-optimized introsort
    std::sort(arr.begin(), arr.end());
}