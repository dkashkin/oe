// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

// EVOLVE-BLOCK-START
pub fn adaptive_sort<T: Ord + Clone>(a: &mut [T]) {
    let n = a.len();
    if n < 2 { return }
    let (mut s, mut r) = (true, true);
    for i in 1..n {
        if a[i-1] > a[i] { s = false }
        if a[i-1] < a[i] { r = false }
        if !s && !r { break }
    }
    if s { return }
    if r { a.reverse(); return }
    qs(a);
}

fn qs<T: Ord + Clone>(a: &mut [T]) {
    let n = a.len();
    if n < 24 { return is(a) }
    let (m, l) = (n/2, n-1);
    if a[0] > a[m] { a.swap(0, m) }
    if a[0] > a[l] { a.swap(0, l) }
    if a[m] > a[l] { a.swap(m, l) }
    let p = a[m].clone();
    let (mut lt, mut gt, mut i) = (if a[0] < p { 1 } else { 0 }, if a[l] > p { l } else { n }, 1);
    while i < gt {
        if a[i] < p { a.swap(lt, i); lt += 1; i += 1 }
        else if a[i] > p { gt -= 1; a.swap(i, gt) }
        else { i += 1 }
    }
    let (left, rest) = a.split_at_mut(lt);
    qs(left);
    qs(&mut rest[gt-lt..]);
}
fn is<T: Ord + Clone>(a: &mut [T]) {
    for i in 1..a.len() {
        if a[i] < a[i-1] {
            let v = a[i].clone();
            let mut j = i;
            while j > 0 && a[j-1] > v { a[j] = a[j-1].clone(); j -= 1 }
            a[j] = v;
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