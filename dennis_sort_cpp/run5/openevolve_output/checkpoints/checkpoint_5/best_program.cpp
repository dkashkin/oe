#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    if (!std::is_sorted(arr.begin(), arr.end())) {
        std::sort(arr.begin(), arr.end());
    }
}