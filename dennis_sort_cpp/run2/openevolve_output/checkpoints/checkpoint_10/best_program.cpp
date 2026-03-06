// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Hybrid 3-way QuickSort: Optimizes for duplicates, nearly sorted, and small arrays
void insertion_sort(std::vector<int>& a, int l, int r) {
    for (int i = l + 1; i <= r; ++i) {
        int v = a[i], j = i;
        while (j > l && a[j - 1] > v) { a[j] = a[j - 1]; j--; }
        a[j] = v;
    }
}

void quicksort(std::vector<int>& a, int l, int r) {
    while (r - l > 16) {
        int m = l + (r - l) / 2;
        if (a[m] < a[l]) std::swap(a[l], a[m]);
        if (a[r] < a[l]) std::swap(a[l], a[r]);
        if (a[r] < a[m]) std::swap(a[m], a[r]);
        int p = a[m], lt = l, gt = r, i = l;
        while (i <= gt) {
            if (a[i] < p) std::swap(a[lt++], a[i++]);
            else if (a[i] > p) std::swap(a[i], a[gt--]);
            else i++;
        }
        if (lt - l < r - gt) { quicksort(a, l, lt - 1); l = gt + 1; }
        else { quicksort(a, gt + 1, r); r = lt - 1; }
    }
    insertion_sort(a, l, r);
}

void adaptive_sort(std::vector<int>& arr) {
    if (arr.size() > 1) quicksort(arr, 0, static_cast<int>(arr.size()) - 1);
}
// EVOLVE-BLOCK-END
