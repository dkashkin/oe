// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

void adaptive_sort(std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return;
    }
    quicksort(arr, 0, static_cast<int>(arr.size()) - 1);
}

void quicksort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot_index = partition(arr, low, high);
        if (pivot_index > 0) {
            quicksort(arr, low, pivot_index - 1);
        }
        quicksort(arr, pivot_index + 1, high);
    }
}

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            std::swap(arr[i], arr[j]);
            ++i;
        }
    }
    std::swap(arr[i], arr[high]);
    return i;
}

// Helper function to detect if array is nearly sorted
bool is_nearly_sorted(const std::vector<int>& arr, double threshold) {
    if (arr.size() <= 1) {
        return true;
    }
    int inversions = 0;
    double max_inversions = (static_cast<double>(arr.size()) * (arr.size() - 1) / 2.0) * threshold;
    for (size_t i = 0; i < arr.size() - 1; ++i) {
        for (size_t j = i + 1; j < arr.size(); ++j) {
            if (arr[i] > arr[j]) {
                ++inversions;
                if (static_cast<double>(inversions) > max_inversions) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Helper function for insertion sort (useful for small arrays)
void insertion_sort(std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        int j = static_cast<int>(i);
        while (j > 0 && arr[j - 1] > arr[j]) {
            std::swap(arr[j], arr[j - 1]);
            --j;
        }
    }
}
// EVOLVE-BLOCK-END
