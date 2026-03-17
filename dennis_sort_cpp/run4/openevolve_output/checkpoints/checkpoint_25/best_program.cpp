#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    const size_t n = arr.size();
    if (n < 32) {
        std::sort(arr.begin(), arr.end());
        return;
    }
    auto it = std::is_sorted_until(arr.begin(), arr.end());
    if (it == arr.end()) return;
    auto ir = arr.begin() + 1;
    while (ir != arr.end() && ir[-1] >= ir[0]) ++ir;
    if (ir > it) {
        std::reverse(arr.begin(), ir);
        it = ir;
        if (it == arr.end()) return;
    }
    if (arr.end() - it < 64 || it - arr.begin() > (ptrdiff_t)(n / 2)) {
        std::sort(it, arr.end());
        std::inplace_merge(arr.begin(), it, arr.end());
    } else {
        std::sort(arr.begin(), arr.end());
    }
}