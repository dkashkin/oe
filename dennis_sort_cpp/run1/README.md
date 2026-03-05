# C++ Adaptive Sorting Evolution

This example demonstrates how to use OpenEvolve with C++ to evolve adaptive sorting algorithms that optimize their behavior based on input data characteristics.

## Files

- `initial_program.cpp`: Starting C++ implementation with basic quicksort
- `evaluator.py`: Python evaluator that compiles and benchmarks C++ code
- `sort_test/main.cpp`: Fixed benchmark harness (not evolved)
- `config.yaml`: Configuration for the evolution process

## Prerequisites

- **g++** with C++17 support (GCC 7+, typically pre-installed on Linux)
- **Python 3.10+** with OpenEvolve installed

No external C++ libraries are needed — only the C++ standard library is used.

## Usage

```bash
python ../../openevolve-run.py initial_program.cpp evaluator.py --config config.yaml --iterations 10
```
