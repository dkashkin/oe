#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    int n = arr.size(), i = 1;
    if (n < 2) return;
    while (i < n && arr[i] <= arr[i-1]) ++i;
    if (arr[0] > arr[i-1]) std::reverse(arr.begin(), arr.begin() + i);
    while (i < n && arr[i-1] <= arr[i]) ++i;
    if (i == n) return;
    int mn = arr[0], mx = mn;
    for (int x : arr) { mn = std::min(mn, x); mx = std::max(mx, x); }
    long long r = (long long)mx - mn;
    if (r < 2000000 && r < n * 32LL) {
        std::vector<int> c(r + 1, 0);
        for (int x : arr) c[x - mn]++;
        auto it = arr.begin();
        for (int j = 0; j <= (int)r; ++j)
            if (int v = c[j]) it = std::fill_n(it, v, j + mn);
        return;
    }
    if (i * 4 > n || n - i < 80) {
        std::sort(arr.begin() + i, arr.end());
        if (arr[i] < arr[i-1]) std::inplace_merge(arr.begin(), arr.begin() + i, arr.end());
    } else std::sort(arr.begin(), arr.end());
}