// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Optimized Hybrid 3-way QuickSort: Median-of-three, Dijkstra partitioning, and inlined insertion sort
void quicksort(std::vector<int>& a, int l, int r) {
    while (r - l > 20) {
        int lt = l, gt = r, i = l, m = l + (r - l) / 2;
        if (a[m] < a[l]) std::swap(a[l], a[m]);
        if (a[r] < a[l]) std::swap(a[l], a[r]);
        if (a[r] < a[m]) std::swap(a[m], a[r]);
        int p = a[m];
        while (i <= gt) {
            if (a[i] < p) std::swap(a[lt++], a[i++]);
            else if (a[i] > p) std::swap(a[i], a[gt--]);
            else i++;
        }
        if (lt - l < r - gt) { quicksort(a, l, lt - 1); l = gt + 1; }
        else { quicksort(a, gt + 1, r); r = lt - 1; }
    }
    for (int i = l + 1; i <= r; ++i) {
        int v = a[i], j = i;
        if (v < a[j - 1]) {
            while (j > l && a[j - 1] > v) { a[j] = a[j - 1]; j--; }
            a[j] = v;
        }
    }
}

void adaptive_sort(std::vector<int>& arr) {
    int n = arr.size(), i = 1;
    if (n < 2) return;
    while (i < n && arr[i - 1] <= arr[i]) ++i;
    if (i == n) return;
    if (i == 1) {
        while (i < n && arr[i - 1] >= arr[i]) ++i;
        if (i == n) {
            for (int l = 0, r = n - 1; l < r; ) std::swap(arr[l++], arr[r--]);
            return;
        }
    }
    quicksort(arr, 0, n - 1);
}
// EVOLVE-BLOCK-END
