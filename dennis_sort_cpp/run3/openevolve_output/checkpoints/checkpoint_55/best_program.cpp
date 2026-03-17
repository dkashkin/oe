// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

void adaptive_sort(std::vector<int>& a) {
    int n = a.size(); if (n < 2) return;
    bool s = 1, r = 1;
    for (int i = 1; i < n && (s || r); ++i)
        if (a[i] < a[i-1]) s = 0; else if (a[i] > a[i-1]) r = 0;
    if (s) return;
    if (r) { for (int i = 0; i < n / 2; ++i) std::swap(a[i], a[n - 1 - i]); return; }
    quicksort(a, 0, n - 1);
}

void quicksort(std::vector<int>& a, int l, int h) {
    while (l < h) {
        if (h - l < 22) {
            for (int i = l + 1; i <= h; ++i) {
                int v = a[i], j = i;
                while (j > l && a[j-1] > v) { a[j] = a[j-1]; --j; }
                a[j] = v;
            } return;
        }
        int m = l+(h-l)/2, lt = l, gt = h, i = l, v;
        if (a[m] < a[l]) std::swap(a[l], a[m]);
        if (a[h] < a[l]) std::swap(a[l], a[h]);
        if (a[h] < a[m]) std::swap(a[m], a[h]);
        v = a[m];
        while (i <= gt) {
            if (a[i] < v) std::swap(a[lt++], a[i++]);
            else if (a[i] > v) {
                while (gt > i && a[gt] > v) --gt;
                std::swap(a[i], a[gt--]);
            } else ++i;
        }
        if (lt - l < h - gt) { quicksort(a, l, lt - 1); l = gt + 1; }
        else { quicksort(a, gt + 1, h); h = lt - 1; }
    }
}