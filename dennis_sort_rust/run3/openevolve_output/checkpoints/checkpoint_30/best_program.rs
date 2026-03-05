// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

use std::cmp::Ordering;

// EVOLVE-BLOCK-START
// Adaptive hybrid Quicksort: Median-of-three pivot selection, insertion sort
// for small partitions, and O(N) detection of sorted/reverse-sorted data.

pub fn adaptive_sort<T: Ord + Clone>(mut arr: &mut [T]) {
    while arr.len() > 24 {
        let n = arr.len();
        let (mut asc, mut desc) = (true, true);
        for i in 1..n {
            if arr[i-1] > arr[i] { asc = false; }
            else if arr[i-1] < arr[i] { desc = false; }
            if !asc && !desc { break; }
        }
        if asc { return; }
        if desc { return arr.reverse(); }

        let (m, h) = (n / 2, n - 1);
        if arr[m] < arr[0] { arr.swap(0, m); }
        if arr[h] < arr[0] { arr.swap(0, h); }
        if arr[h] < arr[m] { arr.swap(m, h); }
        arr.swap(m, h);
        
        let (pivot, mut i) = (arr[h].clone(), 0);
        for j in 0..h {
            if arr[j] <= pivot {
                arr.swap(i, j);
                i += 1;
            }
        }
        arr.swap(i, h);
        
        if i < n / 2 {
            adaptive_sort(&mut arr[..i]);
            arr = &mut arr[i + 1..];
        } else {
            adaptive_sort(&mut arr[i + 1..]);
            arr = &mut arr[..i];
        }
    }
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