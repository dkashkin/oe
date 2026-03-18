#include <vector>
#include <algorithm>

void adaptive_sort(std::vector<int>& arr) {
    if (arr.size() < 2) return;
    
    // Check if already sorted
    if (std::is_sorted(arr.begin(), arr.end())) {
        return;
    }
    
    // Check if reverse sorted
    if (std::is_sorted(arr.rbegin(), arr.rend())) {
        std::reverse(arr.begin(), arr.end());
    } else {
        // Fallback for random or mostly unsorted data
        std::sort(arr.begin(), arr.end());
    }
}