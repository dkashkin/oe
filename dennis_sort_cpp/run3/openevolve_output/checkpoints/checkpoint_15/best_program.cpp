void quicksort(std::vector<int>& v, int l, int h) {
    if (h - l < 32) {
        for (int i = l + 1, j, t; i <= h; i++) {
            for (t = v[i], j = i; j > l && v[j - 1] > t; j--) v[j] = v[j - 1];
            v[j] = t;
        }
        return;
    }
    int m = l + (h - l) / 2, i = l, j = h, k = l;
    if (v[m] < v[l]) std::swap(v[l], v[m]);
    if (v[h] < v[l]) std::swap(v[l], v[h]);
    if (v[h] < v[m]) std::swap(v[m], v[h]);
    int p = v[m];
    while (k <= j) {
        if (v[k] < p) std::swap(v[i++], v[k++]);
        else if (v[k] > p) std::swap(v[j--], v[k]);
        else k++;
    }
    quicksort(v, l, i - 1); quicksort(v, j + 1, h);
}

void adaptive_sort(std::vector<int>& v) {
    int n = v.size();
    if (n < 2) return;
    bool a = 1, d = 1;
    for (int i = 1; i < n; i++) {
        if (v[i] < v[i - 1]) a = 0;
        if (v[i] > v[i - 1]) d = 0;
        if (!a && !d) { quicksort(v, 0, n - 1); return; }
    }
    if (d) for (int i = 0; i < n / 2; i++) std::swap(v[i], v[n - 1 - i]);
}