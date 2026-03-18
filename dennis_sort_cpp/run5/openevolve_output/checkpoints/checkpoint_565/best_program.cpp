#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& v) {
    size_t n = v.size(), i = 1;
    if (n < 2) return;
    while (i < n && v[i] == v[i-1]) i++;
    if (i < n && v[i] < v[i-1]) {
        while (i < n && v[i] <= v[i-1]) i++;
        std::reverse(v.begin(), v.begin() + i);
    }
    while (i < n && v[i-1] <= v[i]) i++;
    if (i == n) return;
    int mn = v[0], mx = v[0];
    for (int x : v) { if (x < mn) mn = x; if (x > mx) mx = x; }
    long long r = (long long)mx - mn;
    if (r < 2000000 && r < (long long)n * 32) {
        std::vector<int> c(r + 1, 0);
        for (int x : v) c[x - mn]++;
        auto it = v.begin();
        for (int j = 0; j <= (int)r; ++j)
            if (int cnt = c[j]) { std::fill_n(it, cnt, j + mn); it += cnt; }
        return;
    }
    if (i * 4 > n || n - i < 80) {
        std::sort(v.begin() + i, v.end());
        if (v[i] < v[i-1]) std::inplace_merge(v.begin(), v.begin() + i, v.end());
    } else std::sort(v.begin(), v.end());
}