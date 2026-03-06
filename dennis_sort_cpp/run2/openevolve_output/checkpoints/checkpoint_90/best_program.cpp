// Adaptive Sorting Algorithm Implementation
// EVOLVE-BLOCK-START
void quicksort(std::vector<int>& a, int l, int h) {
    while (h - l > 32) {
        int m = l + (h - l) / 2, i = l, j = l, k = h;
        if (a[m] < a[l]) std::swap(a[m], a[l]);
        if (a[h] < a[l]) std::swap(a[h], a[l]);
        if (a[h] < a[m]) std::swap(a[h], a[m]);
        int p = a[m];
        while (j <= k) {
            int r = a[j];
            if (r < p) { a[j++] = a[i]; a[i++] = r; }
            else if (p < r) { a[j] = a[k]; a[k--] = r; }
            else j++;
        }
        if (i - l < h - k) { quicksort(a, l, i - 1); l = k + 1; }
        else { quicksort(a, k + 1, h); h = i - 1; }
    }
    for (int i = l + 1; i <= h; ++i) if (a[i] < a[l]) std::swap(a[i], a[l]);
    for (int i = l + 2; i <= h; ++i) {
        int v = a[i], j = i - 1;
        while (a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

void adaptive_sort(std::vector<int>& arr) {
    int n = (int)arr.size();
    if (n < 2) return;
    int i = 1;
    for (; i < n; ++i) if (arr[i] < arr[i - 1]) break;
    if (i == n) return;
    for (i = 1; i < n; ++i) if (arr[i] > arr[i - 1]) break;
    if (i == n) {
        for (int k = 0; k < n / 2; ++k) std::swap(arr[k], arr[n - 1 - k]);
        return;
    }
    quicksort(arr, 0, n - 1);
}
// EVOLVE-BLOCK-END
