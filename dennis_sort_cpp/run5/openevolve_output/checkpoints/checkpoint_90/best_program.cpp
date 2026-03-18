#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    const size_t n = arr.size();
    if (n < 2) return;

    bool a = 1, d = 1;
    for (size_t i = 1; i < n; ++i) {
        if (arr[i-1] > arr[i]) a = 0;
        else if (arr[i-1] < arr[i]) d = 0;
        if (!(a || d)) break;
    }
    if (a) return;
    if (d) { std::reverse(arr.begin(), arr.end()); return; }

    int mn = arr[0], mx = arr[0];
    for (int x : arr) {
        mn = std::min(mn, x);
        mx = std::max(mx, x);
    }

    const long long r = (long long)mx - mn;
    if (r < 2000000 && r < (long long)n * 24) {
        std::vector<int> c(r + 1, 0);
        for (int x : arr) c[x - mn]++;
        auto it = arr.begin();
        for (int i = 0; i <= (int)r; ++i)
            if (int cnt = c[i]) {
                std::fill_n(it, cnt, i + mn);
                it += cnt;
            }
        return;
    }
    std::sort(arr.begin(), arr.end());
}