# NVIDIA Nemotron Model Reasoning Challenge

**Timeline**: March 16 – June 15, 2026 (~3 months)
**Platform**: Kaggle | **Infrastructure**: Google Cloud G4 VMs
**Competition URL**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

---

## What is this?

This is **NOT** a solve-math-problems competition like AIMO. It's a **model improvement challenge**: take NVIDIA's **Nemotron 3 Nano** (a 30B total / 3B active parameter Mixture-of-Experts model) and improve its reasoning accuracy on a new NVIDIA-developed benchmark. You're improving a model, not just running inference.

---

## The Model: Nemotron 3 Nano

- **Architecture**: Hybrid Mamba-Transformer MoE
- **Size**: 30B total params, 3B active (very compute-efficient)
- **Context**: 1M token native context window
- **Target hardware**: DGX Spark, H100, B200 — and G4 VMs in this competition

---

## Evaluation Metric

- **pass@1** — average accuracy across 64 generations per problem
- **maj@64** — majority voting across 64 generations
- Benchmark: A **new reasoning benchmark** developed by NVIDIA (evaluated server-side on Kaggle)

---

## Allowed Techniques (any or all)

| Technique | What it means |
|---|---|
| **Prompt engineering** | System prompts, few-shot, CoT formatting |
| **Synthetic data generation** | Generate training data using larger models |
| **Data filtering/curation** | Select high-quality reasoning traces |
| **Lightweight fine-tuning** | SFT, LoRA/QLoRA on reasoning data |
| **Reinforcement learning** | GRPO, PPO, REINFORCE on reward signal |

---

## Roadmap

### Phase 0 — Setup & Baseline (Week 1)

#### Step 1 — Understand the Model First (Before Any Code)

The model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`:

```
Architecture: Hybrid Mamba2-Transformer MoE
Total params: 30B  |  Active params: 3.5B per token
Layers: 52 total
  - 23 Mamba-2 layers      (fast sequential state tracking)
  - 23 MoE layers           (128 routed experts, only 6 active per token)
  - 6  Attention layers     (GQA, 2 groups — memory efficient)
Context: up to 1M tokens
Training: 25 trillion tokens
```

**Why does architecture matter?** Mamba layers use state-space models (SSMs) — recurrent, not attention-based. This means:
- Inference is fast at long contexts (linear, not quadratic like attention)
- Not all Transformer optimizations apply
- vLLM >= 0.12.0 is required (older versions don't support Mamba)

The model has **two modes**:

| Mode | When to use | Params |
|---|---|---|
| `enable_thinking=True` (default) | Reasoning tasks, math | `temp=1.0, top_p=1.0, max_new_tokens=10000` |
| `enable_thinking=False` | Fast non-reasoning tasks | `do_sample=False, greedy` |

For this competition: **always use thinking mode ON**.

---

#### Step 2 — Hardware Reality Check

The competition runs on **Google Cloud G4 VMs** with **NVIDIA L4 GPUs** (24 GB VRAM each).

The model in BF16 = ~60 GB → **doesn't fit on one L4**.

Options:

| Option | Model Variant | VRAM | Notes |
|---|---|---|---|
| **Recommended** | FP8 variant | ~15 GB | Fits on 1× L4 |
| Multi-GPU | BF16 variant | ~60 GB | 3× L4, tensor parallel |
| GGUF | unsloth GGUF | Variable | CPU/mixed, slower |

**Start with FP8 on a single L4. Get it working first, optimize later.**

---

#### Step 3 — Environment Setup (Kaggle Notebook)

```python
# Install dependencies
!pip install -U "vllm>=0.12.0" transformers accelerate compressed-tensors

# Download the custom reasoning parser (required for vLLM)
!wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

# Check GPU
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
```

---

#### Step 4 — Load the Model (Two Approaches)

**Option A: Transformers (simpler, slower — good for initial testing)**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"  # FP8 fits on 1x L4

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)
```

**Option B: vLLM server (production-grade — required for maj@64)**

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --served-model-name model \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --trust-remote-code \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3 \
  --port 8000
```

**Why vLLM matters**: The metric is `maj@64` — you need 64 generations per problem. vLLM's continuous batching makes this feasible; plain Transformers would be too slow.

---

#### Step 5 — Understand the Benchmark Format

Check the competition's Data tab on Kaggle for exact format. A basic inference function:

```python
def solve_problem(problem_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math reasoning assistant. Think step by step."
        },
        {
            "role": "user",
            "content": problem_text
        }
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        tokenized,
        max_new_tokens=10000,
        temperature=1.0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0][tokenized.shape[1]:], skip_special_tokens=True)
```

---

#### Step 6 — Baseline Evaluation with NeMo Evaluator

NVIDIA open-sourced their exact evaluation pipeline:

```bash
pip install nemo-evaluator-launcher

export HF_TOKEN="your-hf-token"

# Quick test: 10 samples from AIME 2025 (most relevant benchmark)
nemo-evaluator-launcher run \
  --config local_nvidia_nemotron_3_nano_30b_a3b.yaml \
  -t ns_aime2025 \
  -o evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

**Baseline numbers to beat** (from NVIDIA's model card):

| Benchmark | Baseline Score |
|---|---|
| AIME 2025 (with tools) | 89.1% |
| MMLU-Pro | 78.3% |
| GPQA (grad-level science) | 73.0% |
| LiveCodeBench | 68.3% |

---

#### Step 7 — Quick Wins Already Available in Phase 0

Phase 0 isn't just setup — you can already make meaningful improvements:

1. **Reasoning budget**: Test `max_new_tokens` at 1000 / 5000 / 10000. More thinking = higher accuracy, but slower.
2. **Sampling params**: `temp=1.0, top_p=1.0` is recommended, but test `temp=0.7` — lower temperature sometimes helps for math.
3. **System prompt**: Default vs "Think step by step" vs math-specific prompts can shift accuracy by 2–5%.
4. **Thinking mode**: Always `enable_thinking=True` for math reasoning.

---

#### Phase 0 Checklist

```
[ ] Accept competition on Kaggle, check Data tab for benchmark format
[ ] Spin up Kaggle notebook with G4/L4 GPU accelerator
[ ] Install vLLM >= 0.12.0 + download reasoning parser
[ ] Load nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 (fits on 1× L4)
[ ] Run 1–2 sample problems with enable_thinking=True → verify output format
[ ] Install nemo-evaluator-launcher, run ns_aime2025 with limit_samples=10
[ ] Record baseline score → this is your floor for all future phases
[ ] Experiment: 3 system prompts × 2 temperature settings → pick best combo
[ ] Note inference speed (tokens/sec) → critical for time budget planning
```

### Phase 1 — Prompt Engineering (Week 1–2)
- Try CoT, TIR (tool-integrated reasoning with Python), structured output formats
- Best prompt format for Nemotron's MoE architecture
- Self-consistency / maj@k voting to boost pass@1

### Phase 2 — Synthetic Data (Week 2–4)
- Use larger models (DeepSeek-R1, QwQ-32B, or Nemotron Ultra) to generate reasoning traces
- Filter by correctness (process reward model or just answer verification)
- Build a curated SFT dataset

### Phase 3 — Fine-tuning (Week 3–6)
- SFT on synthetic data using LoRA/QLoRA (fits on G4)
- Compare TIR vs pure CoT traces
- Use NVIDIA's NeMo-Skills pipeline (open source, used by AIMO2 winners)

### Phase 4 — RL Post-training (Week 5–8)
- GRPO (Group Relative Policy Optimization) — simpler than PPO, no value model
- Reward = binary correctness on math problems
- Iterate with small batches on G4

### Phase 5 — Ensemble & Submit (Week 8–12)
- Combine CoT + TIR reasoning paths
- Early stopping convergence (stop when 4+ generations agree, like AIMO2 winners)
- Final submission optimization

---

## Key Resources

| Resource | Link |
|---|---|
| NeMo-Skills pipeline (AIMO2 winners) | https://github.com/NVIDIA/NeMo-Skills |
| OpenMathReasoning dataset (5.68M examples) | https://huggingface.co/datasets/nvidia/OpenMathReasoning |
| Reference model (AIMO2 winner, 14B) | https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle |
| NVIDIA blog on AIMO2 win | https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/ |
| Nemotron 3 Super tech blog | https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/ |

---

## Connection to AIMO Project

This competition is directly adjacent to the AIMO project. AIMO notebooks already cover:
- CoT baseline → Phase 1 here
- TIR loop → Phase 1 here
- Majority voting → core eval technique here

New skills to add: **fine-tuning + RL post-training on a MoE model**.

---

## Key Insights from AIMO2 Winners (NVIDIA NemoSkills Team)

- Fine-tuned Qwen2.5-14B-Base on 5.68M synthetic reasoning traces
- Synthetic data generated by DeepSeek-R1 and QwQ-32B (knowledge distillation)
- **Parallel reasoning with early-stopping**: stop when 4+ generations converge on same answer
- **FP8 quantization via TensorRT-LLM**: 1.5x speedup over FP16
- **ReDrafter speculative decoding**: additional 1.8x speedup
- Hybrid reasoning: natural language CoT + Python code execution (TIR)
- Result: 34/50 on hidden test set using 4x L4 GPUs

## Phases Status

| Phase | Status | Notes |
|---|---|---|
| 0. Setup & Baseline | Pending | — |
| 1. Prompt Engineering | Pending | — |
| 2. Synthetic Data | Pending | — |
| 3. Fine-tuning | Pending | — |
| 4. RL Post-training | Pending | — |
| 5. Ensemble & Submit | Pending | — |
