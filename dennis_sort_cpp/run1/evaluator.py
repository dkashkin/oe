"""
Evaluator for C++ adaptive sorting example
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from openevolve.evaluation_result import EvaluationResult
import logging
import os
import shutil

THIS_FILE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger("examples.cpp_adaptive_sort.evaluator")


def evaluate(program_path: str) -> EvaluationResult:
    result = asyncio.run(_evaluate(program_path))
    if "error" in result.artifacts:
        logger.error(f"Error evaluating program: {result.artifacts['error']}")
        if "stderr" in result.artifacts:
            logger.error(f"Stderr: {result.artifacts['stderr']}")
        if "stdout" in result.artifacts:
            logger.error(f"Stdout: {result.artifacts['stdout']}")
    return result


async def _evaluate(program_path: str) -> EvaluationResult:
    """
    Evaluate a C++ sorting algorithm implementation.

    Tests the algorithm on various data patterns to measure:
    - Correctness
    - Performance (speed)
    - Adaptability to different data patterns
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the evolving program as sort_impl.cpp
            shutil.copy2(program_path, temp_path / "sort_impl.cpp")

            # Copy the fixed benchmark harness
            main_source = THIS_FILE_DIR / "sort_test" / "main.cpp"
            shutil.copy2(main_source, temp_path / "main.cpp")

            # Compile with g++
            binary_path = temp_path / "sort_test"
            build_result = subprocess.run(
                [
                    "g++", "-std=c++17", "-O2", "-o", str(binary_path),
                    str(temp_path / "main.cpp"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if build_result.returncode != 0:
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 0.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={
                        "error": "Compilation failed",
                        "stderr": build_result.stderr,
                        "stdout": build_result.stdout,
                    },
                )

            # Run the benchmark
            run_result = subprocess.run(
                [str(binary_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if run_result.returncode != 0:
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={"error": "Runtime error", "stderr": run_result.stderr},
                )

            # Parse JSON output
            try:
                output = run_result.stdout
                start = output.find("{")
                end = output.rfind("}") + 1
                json_str = output[start:end]

                results = json.loads(json_str)

                correctness = results["correctness"]
                performance = results["performance_score"]
                adaptability = results["adaptability_score"]

                if correctness < 1.0:
                    overall_score = 0.0
                else:
                    overall_score = 0.6 * performance + 0.4 * adaptability

                return EvaluationResult(
                    metrics={
                        "score": overall_score,
                        "combined_score": overall_score,
                        "compile_success": 1.0,
                        "correctness": correctness,
                        "performance_score": performance,
                        "adaptability_score": adaptability,
                        "avg_time": results["avg_time"],
                    },
                    artifacts={
                        "times": results["times"],
                        "all_correct": results["all_correct"],
                        "build_output": build_result.stdout,
                    },
                )

            except (json.JSONDecodeError, KeyError) as e:
                return EvaluationResult(
                    metrics={
                        "score": 0.0,
                        "compile_success": 1.0,
                        "correctness": 0.0,
                        "performance_score": 0.0,
                        "adaptability_score": 0.0,
                    },
                    artifacts={
                        "error": f"Failed to parse results: {str(e)}",
                        "stdout": run_result.stdout,
                    },
                )

    except subprocess.TimeoutExpired:
        return EvaluationResult(
            metrics={
                "score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0,
            },
            artifacts={"error": "Timeout during evaluation"},
        )
    except Exception as e:
        return EvaluationResult(
            metrics={
                "score": 0.0,
                "compile_success": 0.0,
                "correctness": 0.0,
                "performance_score": 0.0,
                "adaptability_score": 0.0,
            },
            artifacts={"error": str(e), "type": "evaluation_error"},
        )


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = evaluate(sys.argv[1])
        print(f"Score: {result.metrics['score']:.4f}")
        print(f"Correctness: {result.metrics['correctness']:.4f}")
        print(f"Performance: {result.metrics['performance_score']:.4f}")
        print(f"Adaptability: {result.metrics['adaptability_score']:.4f}")
