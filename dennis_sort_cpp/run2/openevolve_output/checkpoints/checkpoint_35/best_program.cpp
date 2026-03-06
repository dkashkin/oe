// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

void quicksort(std::vector<int>& a, int l, int h) {
    while (l < h) {
        if (h - l < 24) {
            for (int i = l + 1; i <= h; i++) {
                int v = a[i], j = i - 1;
                while (j >= l && a[j] > v) { a[j + 1] = a[j]; j--; }
                a[j + 1] = v;
            }
            return;
        }
        int m = l + (h - l) / 2, i = l, j = l, k = h, v;
        if (a[m] < a[l]) std::swap(a[m], a[l]);
        if (a[h] < a[l]) std::swap(a[h], a[l]);
        if (a[h] < a[m]) std::swap(a[h], a[m]);
        v = a[m];
        while (j <= k) {
            if (a[j] < v) std::swap(a[i++], a[j++]);
            else if (a[j] > v) std::swap(a[j], a[k--]);
            else j++;
        }
        if (i - l < h - k) { quicksort(a, l, i - 1); l = k + 1; }
        else { quicksort(a, k + 1, h); h = i - 1; }
    }
}

void adaptive_sort(std::vector<int>& arr) {
    int n = (int)arr.size();
    if (n < 2) return;
    bool asc = 1, dsc = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) asc = 0;
        if (arr[i] > arr[i - 1]) dsc = 0;
        if (!asc && !dsc) break;
    }
    if (asc) return;
    if (dsc) {
        for (int i = 0; i < n / 2; i++) std::swap(arr[i], arr[n - 1 - i]);
        return;
    }
    quicksort(arr, 0, n - 1);
}
// EVOLVE-BLOCK-END
