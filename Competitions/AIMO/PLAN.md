# AIMO Progress Prize 3 — Learning Plan

## Competition
- **Name:** AI Mathematical Olympiad - Progress Prize 3
- **URL:** https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- **Task:** Solve 110 olympiad-level math problems; output a 5-digit integer answer per problem
- **Topics:** Algebra, Combinatorics, Geometry, Number Theory
- **Evaluation:** Accuracy (# problems correct) on public + private test sets
- **Prize:** $2.2 million total
- **Key rule:** Solution must be open-source

## Core Insight
The winning pattern is: **Tool-Integrated Reasoning (TIR) + Self-Consistency**

```
Problem → LLM writes reasoning + Python code → Execute code → Feed output back → LLM self-corrects
                ↑_______________________________________________________↓
                              (loop until answer produced)

Run this N=48 times in parallel → Majority vote → Final answer
```

---

## Phase 0 — Why is math hard for AI?
**Goal:** Understand the problem before touching code.
**Notebook:** `01_cot_baseline.ipynb` (top section)
**Concepts:**
- LLMs as next-token predictors, not calculators
- Why hallucination is catastrophic for math
- Why code execution is the key unlock

## Phase 1 — Chain-of-Thought Prompting
**Goal:** Build and measure a CoT baseline.
**Notebook:** `01_cot_baseline.ipynb`
**Concepts:** CoT, zero-shot vs few-shot, temperature, sampling
**Model:** `Qwen/Qwen2.5-Math-7B-Instruct`
**Status:** [x] Scaffolded

## Phase 2 — Tool-Integrated Reasoning (TIR) Loop
**Goal:** Build the agentic inference loop: LLM + code execution + self-correction.
**Notebook:** `02_tir_loop.ipynb`
**Concepts:** Agentic loops, sympy, code sandboxing, iterative TIR
**Key functions:** `extract_python_blocks`, `safe_execute`, `extract_boxed_answer`, `run_tir_loop`
**Status:** [x] Scaffolded

## Phase 3 — Self-Consistency + Majority Voting
**Goal:** Sample N solutions, majority-vote the answer. Quantify variance reduction.
**Notebook:** `03_majority_voting.ipynb`
**Concepts:** Monte Carlo sampling, majority vote, GenSelect
**Status:** [ ] Pending

## Phase 4 — Model Selection + Fine-tuning (Optional)
**Goal:** Understand which models are best for math; optionally fine-tune with QLoRA.
**Notebook:** `04_model_selection.ipynb`
**Concepts:** Math-specialized LLMs, SFT, QLoRA, RLVR, NuminaMath dataset
**Status:** [ ] Pending

## Phase 5 — Submission Pipeline
**Goal:** Package everything into a Kaggle-compliant inference notebook.
**Notebook:** `05_submission.ipynb`
**Concepts:** vLLM, quantization, time budgeting, Kaggle GPU limits
**Status:** [ ] Pending

---

## Models to Know

| Model | Size | Strength |
|---|---|---|
| Qwen2.5-Math-7B-Instruct | 7B | Strong math, fast, fits on T4 |
| DeepSeekMath-7B-Instruct | 7B | Strong CoT + TIR |
| Qwen2.5-Math-72B-Instruct | 72B | Best open math model (needs A100) |
| NuminaMath-7B-TIR | 7B | AIMO1 winner's model |

## Key Libraries

| Library | Use |
|---|---|
| `transformers` | Load models |
| `vllm` | Fast batched inference |
| `sympy` | Symbolic math (models use this) |
| `trl` | Fine-tuning (SFT / GRPO) |
| `bitsandbytes` | 4-bit quantization |
| `datasets` | HuggingFace datasets |

## Previous Winners
- **AIMO1:** Project Numina — NuminaMath-7B-TIR, 29/50 problems
- **AIMO2:** Nvidia NemoSkills — GenSelect + TIR, 34/50 problems
