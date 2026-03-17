#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    if (arr.size() < 2) return;
    if (std::is_sorted(arr.begin(), arr.end())) return;
    if (std::is_sorted(arr.rbegin(), arr.rend())) {
        std::reverse(arr.begin(), arr.end());
    } else {
        std::sort(arr.begin(), arr.end());
    }
}