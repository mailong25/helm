#!/bin/bash

# ===============================
#  Run Benchmark
# ===============================

# Set evaluation instance limit and model
export MAX_EVAL_INSTANCES=5
export EVAL_MODEL="openai/gpt-4o"

# ===============================
# Subject Knowledge Test
# ===============================

echo "Running GPQA..."
helm-run --run-entries gpqa:subset=gpqa_main,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - High School Biology..."
# helm-run --run-entries mmlu:subject=high_school_biology,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - College Biology..."
# helm-run --run-entries mmlu:subject=college_biology,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - Abstract Algebra..."
# helm-run --run-entries mmlu:subject=abstract_algebra,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - College Chemistry..."
# helm-run --run-entries mmlu:subject=college_chemistry,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - Computer Security..."
# helm-run --run-entries mmlu:subject=computer_security,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - Econometrics..."
# helm-run --run-entries mmlu:subject=econometrics,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# echo "Running MMLU - US Foreign Policy..."
# helm-run --run-entries mmlu:subject=us_foreign_policy,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# # ===============================
# # Safety Test
# # ===============================

# echo "Running Simple Safety Tests..."
# helm-run --run-entries simple_safety_tests:model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# # ===============================
# # Instruction Following Test
# # ===============================

# echo "Running Instruction Following Evaluation..."
# helm-run --run-entries ifeval:model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# # ===============================
# # Bias/Fairness Test
# # ===============================

# echo "Running BBQ (Bias/Fairness) Evaluation..."
# helm-run --run-entries bbq:subject=all,model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES

# # ===============================
# # Toxicity Test
# # ===============================

# echo "Running Real Toxicity Prompts..."
# helm-run --run-entries real_toxicity_prompts:model=$EVAL_MODEL --suite my-suite --max-eval-instances $MAX_EVAL_INSTANCES -n 1

# echo "All evaluations completed."
