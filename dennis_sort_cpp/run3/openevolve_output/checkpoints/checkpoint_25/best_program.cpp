#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    size_t n = arr.size(), i = 1;
    if (n < 2) return;
    int* d = arr.data();
    while (i < n && d[i] == d[i - 1]) i++;
    if (i == n) return;
    if (d[i] > d[i - 1]) {
        for (i++; i < n; ++i) if (d[i] < d[i - 1]) break;
        if (i == n) return;
    } else {
        for (i++; i < n; ++i) if (d[i] > d[i - 1]) break;
        if (i == n) {
            std::reverse(d, d + n);
            return;
        }
    }
    std::sort(d, d + n);
}