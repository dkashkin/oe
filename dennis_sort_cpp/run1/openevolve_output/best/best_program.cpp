// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Optimized hybrid Quicksort (Hoare partition) with adaptive pre-checks
void quicksort(std::vector<int>& a, int l, int h) {
    while (h - l > 16) {
        int m = l + (h - l) / 2;
        if (a[m] < a[l]) std::swap(a[m], a[l]);
        if (a[h] < a[l]) std::swap(a[h], a[l]);
        if (a[h] < a[m]) std::swap(a[h], a[m]);
        int p = a[m], i = l - 1, j = h + 1;
        while (1) {
            while (a[++i] < p);
            while (a[--j] > p);
            if (i >= j) break;
            std::swap(a[i], a[j]);
        }
        if (j - l < h - j) { quicksort(a, l, j); l = j + 1; }
        else { quicksort(a, j + 1, h); h = j; }
    }
    for (int i = l + 1; i <= h; ++i) {
        int v = a[i], j = i;
        while (j > l && a[j - 1] > v) { a[j] = a[j - 1]; --j; }
        a[j] = v;
    }
}

void adaptive_sort(std::vector<int>& a) {
    int n = static_cast<int>(a.size());
    if (n < 2) return;
    bool s = true, r = true;
    for (int i = 1; i < n; ++i) {
        if (a[i] < a[i - 1]) s = false;
        if (a[i] > a[i - 1]) r = false;
        if (!s && !r) break;
    }
    if (s) return;
    if (r) { for (int i = 0; i < n / 2; ++i) std::swap(a[i], a[n - 1 - i]); return; }
    quicksort(a, 0, n - 1);
}
// EVOLVE-BLOCK-END
