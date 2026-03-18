#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    int n = arr.size(); if (n < 2) return;
    int* v = arr.data(); bool a=1, d=1;
    for (int i=1; i<n; ++i) {
        if (v[i]<v[i-1]) a=0; else if (v[i]>v[i-1]) d=0;
        if (!(a|d)) break;
    }
    if (a) return; if (d) { std::reverse(v, v+n); return; }
    int mn = v[0], mx = v[0];
    for (int x : arr) { mn = std::min(mn, x); mx = std::max(mx, x); }
    long long r = (long long)mx - mn;
    if (r < 2000000 && r < n * 32LL) {
        std::vector<int> c(r + 1, 0);
        for (int x : arr) c[x - mn]++;
        int* p = v;
        for (int i=0; i<=r; ++i) if (int k=c[i]) { std::fill_n(p, k, i+mn); p+=k; }
        return;
    }
    std::sort(arr.begin(), arr.end());
}