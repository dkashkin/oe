#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    size_t n = arr.size();
    if (n < 2) return;
    int* b = arr.data(), *e = b + n;
    int* it = std::is_sorted_until(b, e);
    if (it == e) return;
    if (*b == *(it - 1) && std::is_sorted(it - 1, e, [](int x, int y){ return x > y; })) {
        std::reverse(b, e);
        return;
    }
    std::sort(b, e);
}