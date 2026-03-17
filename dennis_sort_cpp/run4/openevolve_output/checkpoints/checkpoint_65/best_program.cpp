#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    const size_t n = arr.size();
    auto b = arr.begin(), e = arr.end();
    if (n < 64) { std::sort(b, e); return; }
    auto it = std::is_sorted_until(b, e), r = b + 1;
    while (r != e && r[-1] >= r[0]) ++r;
    if (r > it) { std::reverse(b, r); it = r; }
    if (it == e) return;
    if (it - b > (ptrdiff_t)n / 4 || e - it < 128) {
        std::sort(it, e);
        std::inplace_merge(b, it, e);
    } else {
        auto s = e - 1;
        while (s != b && s[-1] <= s[0]) --s;
        if (e - s > (ptrdiff_t)n / 4 || s - b < 128) {
            std::sort(b, s);
            std::inplace_merge(b, s, e);
        } else std::sort(b, e);
    }
}