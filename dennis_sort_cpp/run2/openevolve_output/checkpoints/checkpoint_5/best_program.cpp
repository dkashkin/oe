// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

void quicksort(std::vector<int>& arr, int low, int high);
int partition(std::vector<int>& arr, int low, int high);
void insertion_sort_range(std::vector<int>& arr, int low, int high);

void adaptive_sort(std::vector<int>& arr) {
    if (arr.size() <= 1) return;
    if (arr.size() < 32) {
        insertion_sort_range(arr, 0, static_cast<int>(arr.size()) - 1);
        return;
    }
    quicksort(arr, 0, static_cast<int>(arr.size()) - 1);
}

void insertion_sort(std::vector<int>& arr) {
    if (arr.size() > 1) insertion_sort_range(arr, 0, static_cast<int>(arr.size()) - 1);
}

void quicksort(std::vector<int>& arr, int low, int high) {
    while (low < high) {
        if (high - low < 16) {
            insertion_sort_range(arr, low, high);
            break;
        }
        int p = partition(arr, low, high);
        if (p - low < high - p) {
            quicksort(arr, low, p - 1);
            low = p + 1;
        } else {
            quicksort(arr, p + 1, high);
            high = p - 1;
        }
    }
}

int partition(std::vector<int>& arr, int low, int high) {
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low]) std::swap(arr[mid], arr[low]);
    if (arr[high] < arr[low]) std::swap(arr[high], arr[low]);
    if (arr[high] < arr[mid]) std::swap(arr[high], arr[mid]);
    std::swap(arr[mid], arr[high]);
    int pivot = arr[high], i = low;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) std::swap(arr[i++], arr[j]);
    }
    std::swap(arr[i], arr[high]);
    return i;
}

void insertion_sort_range(std::vector<int>& arr, int low, int high) {
    for (int i = low + 1; i <= high; ++i) {
        int key = arr[i], j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}
// EVOLVE-BLOCK-END
