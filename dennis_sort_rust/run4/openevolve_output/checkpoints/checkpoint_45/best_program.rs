// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

use std::cmp::Ordering;

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) {
    let n = arr.len();
    if n < 16 { if n > 1 { insertion_sort(arr); } return; }
    let (mut inv, mut rev, lim) = (0, 0, n / 12);
    for w in arr.windows(2) {
        if w[0] > w[1] { inv += 1; } else if w[0] < w[1] { rev += 1; }
        if inv > lim && rev > lim { break; }
    }
    if inv <= lim { insertion_sort(arr); }
    else if rev <= lim { arr.reverse(); insertion_sort(arr); }
    else { quicksort(arr, 0, n - 1); }
}

fn quicksort<T: Ord + Clone>(arr: &mut [T], mut l: usize, mut h: usize) {
    while l < h {
        if h - l < 16 { insertion_sort(&mut arr[l..=h]); return; }
        let p = partition(arr, l, h);
        if p - l < h - p { quicksort(arr, l, p); l = p + 1; }
        else { quicksort(arr, p + 1, h); h = p; }
    }
}

fn partition<T: Ord + Clone>(arr: &mut [T], l: usize, h: usize) -> usize {
    let m = l + (h - l) / 2;
    if arr[l] > arr[m] { arr.swap(l, m); }
    if arr[l] > arr[h] { arr.swap(l, h); }
    if arr[m] > arr[h] { arr.swap(m, h); }
    let (v, mut i, mut j) = (arr[m].clone(), l as isize - 1, h as isize + 1);
    loop {
        loop { i += 1; if arr[i as usize] >= v { break; } }
        loop { j -= 1; if arr[j as usize] <= v { break; } }
        if i >= j { return j as usize; }
        arr.swap(i as usize, j as usize);
    }
}

// Helper function for insertion sort (useful for small arrays)
fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}
// EVOLVE-BLOCK-END

// Benchmark function to test the sort implementation
pub fn run_benchmark(test_data: Vec<Vec<i32>>) -> BenchmarkResults {
    let mut results = BenchmarkResults {
        times: Vec::new(),
        correctness: Vec::new(),
        adaptability_score: 0.0,
    };
    
    for data in test_data {
        let mut arr = data.clone();
        let start = std::time::Instant::now();
        
        adaptive_sort(&mut arr);
        
        let elapsed = start.elapsed();
        results.times.push(elapsed.as_secs_f64());
        
        // Check if correctly sorted
        let is_sorted = arr.windows(2).all(|w| w[0] <= w[1]);
        results.correctness.push(is_sorted);
    }
    
    // Calculate adaptability score based on performance variance
    if results.times.len() > 1 {
        let mean_time: f64 = results.times.iter().sum::<f64>() / results.times.len() as f64;
        let variance: f64 = results.times.iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>() / results.times.len() as f64;
        
        // Lower variance means better adaptability
        results.adaptability_score = 1.0 / (1.0 + variance.sqrt());
    }
    
    results
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub times: Vec<f64>,
    pub correctness: Vec<bool>,
    pub adaptability_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_sort() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }
    
    #[test]
    fn test_empty_array() {
        let mut arr: Vec<i32> = vec![];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![]);
    }
    
    #[test]
    fn test_single_element() {
        let mut arr = vec![42];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }
}