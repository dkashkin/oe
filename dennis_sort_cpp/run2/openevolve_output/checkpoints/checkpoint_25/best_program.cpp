// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

void quicksort(std::vector<int>& a, int low, int high);
void insertion_sort_range(std::vector<int>& a, int low, int high);

void adaptive_sort(std::vector<int>& a) {
    int n = a.size();
    if (n <= 1) return;
    bool s = 1, r = 1;
    for (int i = 1; i < n; ++i) {
        if (a[i] < a[i - 1]) s = 0;
        if (a[i] > a[i - 1]) r = 0;
        if (!s && !r) break;
    }
    if (s) return;
    if (r) {
        for (int i = 0, j = n - 1; i < j; ++i, --j) std::swap(a[i], a[j]);
        return;
    }
    if (n < 64) { insertion_sort_range(a, 0, n - 1); return; }
    quicksort(a, 0, n - 1);
}

void insertion_sort(std::vector<int>& a) {
    if (a.size() > 1) insertion_sort_range(a, 0, (int)a.size() - 1);
}

void quicksort(std::vector<int>& a, int low, int high) {
    while (low < high) {
        if (high - low < 24) { insertion_sort_range(a, low, high); break; }
        int m = low + (high - low) / 2;
        if (a[m] < a[low]) std::swap(a[m], a[low]);
        if (a[high] < a[low]) std::swap(a[high], a[low]);
        if (a[high] < a[m]) std::swap(a[high], a[m]);
        std::swap(a[m], a[low]);
        int v = a[low], lt = low, gt = high, i = low;
        while (i <= gt) {
            if (a[i] < v) std::swap(a[i++], a[lt++]);
            else if (a[i] > v) std::swap(a[i], a[gt--]);
            else i++;
        }
        if (lt - low < high - gt) { quicksort(a, low, lt - 1); low = gt + 1; }
        else { quicksort(a, gt + 1, high); high = lt - 1; }
    }
}

void insertion_sort_range(std::vector<int>& a, int l, int h) {
    if (l >= h) return;
    int m = l;
    for (int i = l + 1; i <= h; ++i) if (a[i] < a[m]) m = i;
    std::swap(a[l], a[m]);
    for (int i = l + 2; i <= h; ++i) {
        int k = a[i], j = i - 1;
        while (a[j] > k) { a[j + 1] = a[j]; --j; }
        a[j + 1] = k;
    }
}
// EVOLVE-BLOCK-END
