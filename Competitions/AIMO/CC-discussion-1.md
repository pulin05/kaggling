     AIMO Progress Prize 3 — Learning Plan

  What the competition actually is

  You're building an AI system that can solve olympiad-level math problems (algebra, combinatorics,
  geometry, number theory) and produce a 5-digit integer answer. There are 110 problems total. Guessing is
  essentially impossible — the model must genuinely reason.

  Key constraint: Your solution must be open-source. You submit a Kaggle notebook that runs inference at
  test time.

  ---
  The Core Idea (First Principles)

  The winning pattern from AIMO1 and AIMO2 is always:

  Raw Math Problem
         ↓
    LLM generates reasoning + Python code interleaved
         ↓
    Code gets executed → output fed back to LLM
         ↓
    LLM self-corrects using execution results
         ↓
    Run this 48 times in parallel
         ↓
    Majority vote on the final answers
         ↓
    Submit the most voted answer

  This is called Tool-Integrated Reasoning (TIR) + Self-Consistency.

  ---
  The 6-Phase Learning Plan

  ---
  Phase 0 — Why is math hard for AI? (1–2 days)

  What to learn:
  - How LLMs work at a high level: next-token prediction, not "thinking"
  - Why pure text generation fails at math: hallucination, arithmetic errors, no symbolic grounding
  - Why code execution changes everything: the model doesn't need to compute — it writes Python and lets
  the computer compute

  Concepts:
  - Transformer architecture (intuition only, not the math)
  - Token probability distributions
  - The "stochastic parrot" problem for math

  What to do:
  - Read: How NuminaMath Won the 1st AIMO — the entire post
  - Try: Ask GPT-4 a hard math problem, then ask it to solve it by writing Python code — see the difference

  ---
  Phase 1 — Chain-of-Thought (CoT) Prompting (2–3 days)

  What to learn:
  - What CoT is: "Let's think step by step" — why it improves reasoning
  - Why it works: it forces the model to decompose the problem before committing to an answer
  - Few-shot vs zero-shot CoT
  - Limitations: CoT still hallucinates arithmetic

  Concepts:
  - Prompt engineering
  - In-context learning
  - Temperature and sampling

  What to do:
  - Run a notebook: use a small open model (e.g., Qwen2.5-7B-Instruct or deepseek-math-7b) via Hugging Face
   transformers
  - Prompt it with a few AIME problems using zero-shot CoT: "Solve step by step. Final answer: [number]"
  - Measure accuracy on 10 problems manually

  Key lesson: CoT helps but isn't enough for IMO-level problems.

  ---
  Phase 2 — Tool-Integrated Reasoning / Code Interpreter (3–4 days)

  What to learn:
  - What TIR is: the model writes Python, you execute it, feed the output back, repeat
  - The "TORA format": interleaved natural language + <python>...</python> + [output]... blocks
  - Why self-correction works: the model reads tracebacks and fixes its own bugs
  - Why this is fundamentally different from pure text reasoning

  Concepts:
  - Agentic loops (LLM → tool → LLM cycle)
  - Code sandboxing
  - Python's sympy library (symbolic math — critical for olympiad math)

  What to do:
  - Build a minimal TIR inference loop in a notebook:
    a. Send problem to model
    b. Extract code blocks from output
    c. Execute them with Python exec() in a sandbox
    d. Append output to context, continue generation
    e. Stop when model produces a final answer
  - Solve 5 problems from previous AIMO datasets this way

  Key lesson: Most of the competition win comes from this loop, not from the model itself.

  ---
  Phase 3 — Self-Consistency and Majority Voting (1–2 days)

  What to learn:
  - Why a single LLM run is unreliable: stochastic sampling means different answers each time
  - Self-Consistency: sample N solutions independently, majority-vote the answer
  - Why N=48 (or similar) dramatically reduces variance

  Concepts:
  - Monte Carlo sampling intuition
  - Temperature > 0 for diversity
  - Aggregate functions: majority vote, weighted vote, confidence-based selection

  What to do:
  - Extend your Phase 2 notebook: run the same problem 10 times, collect answers, take the majority
  - Plot answer distribution — see how much variance exists
  - Understand the AIMO2 "GenSelect" idea: use another LLM to pick the best solution from candidates
  instead of just voting

  Key lesson: Inference strategy is as important as the model quality.

  ---
  Phase 4 — Model Selection and (Optional) Fine-tuning (3–5 days)

  What to learn:
  - Why base model choice matters: math pretraining makes a huge difference
  - The key models: DeepSeekMath, Qwen2.5-Math, NuminaMath, Llama-3 variants
  - Fine-tuning: SFT (Supervised Fine-Tuning) on math CoT data
  - Why you may NOT need to fine-tune: good prompting on strong models (Qwen3-120B) can be competitive

  Concepts:
  - Supervised Fine-Tuning (SFT)
  - LoRA / QLoRA (parameter-efficient fine-tuning)
  - RLVR (Reinforcement Learning with Verifiable Rewards) — the new frontier
  - Math datasets: NuminaMath, MATH, AIME archives, AMC archives

  What to do (start here, fine-tune later if compute allows):
  - Try Qwen2.5-Math-7B-Instruct with TIR — it's already math-specialized
  - If you get H100 access: fine-tune with QLoRA on NuminaMath dataset using trl library
  - Track: does fine-tuned model beat base model on your 10-problem test set?

  Compute reality check: Fine-tuning a 7B model needs at least one A100/H100. You can apply for the
  competition's 128×H100 compute grant after your first submission.

  ---
  Phase 5 — Submission Pipeline (2–3 days)

  What to learn:
  - AIMO3 runs inference in a Kaggle notebook with GPU time limits
  - Your notebook must be self-contained: load model weights, run TIR loop, output predictions
  - Time budgeting: 110 problems × (N samples × inference time) must fit in the time limit
  - Quantization: bitsandbytes 4-bit or GPTQ 8-bit to fit larger models in GPU RAM

  Concepts:
  - vLLM for fast batched inference
  - Model quantization (4-bit / 8-bit)
  - asyncio for parallel sampling
  - Kaggle notebook GPU limits

  What to do:
  - Fork the official AIMO 3 Submission Demo notebook
  - Get it running end-to-end with a small model
  - Swap in your better model once the pipeline works

  ---
  Suggested Learning Sequence

  Week 1:  Phase 0 + Phase 1  →  Understand CoT, run first prompting experiments
  Week 2:  Phase 2            →  Build TIR inference loop (the core skill)
  Week 3:  Phase 3            →  Add majority voting, get first leaderboard submission
  Week 4:  Phase 4 (top half) →  Experiment with math-specialized models
  Week 5+: Phase 4 (fine-tune) + Phase 5 → Optimize and iterate

  ---
  Key Libraries to Learn

  ┌──────────────┬─────────────────────────────────────────────────┐
  │   Library    │                     Purpose                     │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ transformers │ Load and run HuggingFace models                 │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ vllm         │ Fast batched LLM inference                      │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ sympy        │ Symbolic math in Python (models use this a lot) │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ trl          │ Fine-tuning with SFT/GRPO                       │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ bitsandbytes │ 4-bit quantization                              │
  ├──────────────┼─────────────────────────────────────────────────┤
  │ datasets     │ HuggingFace dataset loading                     │
  └──────────────┴─────────────────────────────────────────────────┘

  ---
  What Makes This Genuinely Hard

  - IMO-level problems are unsolved by most humans, let alone AI
  - No standard ML tricks apply — this is not about features or gradient descent on tabular data
  - Compute is the bottleneck — more samples = better accuracy, but costs more GPU time
  - The gap is real: best open models score ~34/50; closed models score ~47/50

  You will learn a huge amount just getting to a working TIR loop submission. Winning is a moonshot, but
  the learning is the point.

  ---
  Project Folder Suggestion

  Competitions/
    AIMO-Progress-Prize-3/
      PLAN.md
      notebooks/
        01_cot_baseline.ipynb
        02_tir_loop.ipynb
        03_majority_voting.ipynb
        04_submission.ipynb
      Dataset/
        sample_problems.json

  ---
  Want me to scaffold the project folder and start the first notebook (Phase 0/1 — CoT baseline with a
  small model)?

  Sources:
  - How NuminaMath Won the 1st AIMO Progress Prize
  - Third $2.2 million AIMO Progress Prize Launched
  - AIMO 2 Winning Solution paper
  - AIMO 3 Submission Demo
  - AI Mathematical Olympiad Progress Prize 3 - Kaggle


  -----
  This is primarily an inference engineering project. Fine-tuning is optional.                           
                                                                                                           
  Here's the honest breakdown:                                                                             
                                                                                                           
  What the final objective actually is                                                                     
                                                                                                           
  You submit a Kaggle notebook that receives a math problem and returns an integer. Kaggle runs it against
  110 problems. Your score = how many you get right. That's it.

  The notebook must:
  1. Load a pre-trained model
  2. Run your inference logic (the TIR loop)
  3. Return an integer per problem
  4. Do all of this within 9 hours total

  Is fine-tuning required?

  No. The spectrum looks like this:

  No fine-tuning                                    Fine-tuning
  ──────────────────────────────────────────────────────────────>

  Take a strong math         Fine-tune a smaller      Train from
  model off the shelf   →    model on math data    →   scratch
  (Qwen2.5-Math-72B)         (QLoRA on NuminaMath)    (impractical)

     Most practical               Optional                Skip
     for learning               if compute
                                 available

  The AIMO1 winner did fine-tune — but they were a research team with full H100 access and months of work.
  They built NuminaMath-7B from scratch. Even then, the inference strategy (TIR + self-consistency) was
  responsible for most of the gain over their own baseline.

  The AIMO2 winner also fine-tuned — but on a massive synthetic dataset of 540K problems they generated
  themselves.

  For your learning goals: fine-tuning is Phase 4 and explicitly optional in the plan. You can get a
  competitive submission to the leaderboard purely through inference engineering.

  What the real skill stack is

  90% inference engineering          10% (optional) fine-tuning
  ─────────────────────────          ──────────────────────────
  1. Pick a math model               SFT on NuminaMath dataset
  2. Build the TIR loop              QLoRA to reduce compute cost
  3. Run N=48 samples                RLVR if you want cutting-edge
  4. Majority vote answers
  5. Package for Kaggle gRPC

  The #1 thing that moves scores is steps 2–4, not the model. A mediocre model with a great TIR loop beats
  a great model with no TIR loop every time. That's proven by AIMO1 and AIMO2 results.

  So what are we building, concretely?

  def predict(problem_text: str) -> int:
      candidates = []
      for _ in range(48):                    # sample 48 times
          solution = run_tir_loop(model, problem_text)  # TIR loop
          answer = extract_answer(solution)
          candidates.append(answer)
      return majority_vote(candidates)       # return most common answer

  That function, packaged inside a Kaggle notebook, wired to their gRPC gateway — that is the entire
  deliverable.

  Honest expected outcome

  ┌──────────────────────────┬────────────────────┬──────────────────────────────┐
  │         Approach         │ Score (out of 110) │            Effort            │
  ├──────────────────────────┼────────────────────┼──────────────────────────────┤
  │ CoT only (this notebook) │ 1–5                │ Days                         │
  ├──────────────────────────┼────────────────────┼──────────────────────────────┤
  │ TIR loop, no fine-tuning │ 10–25              │ Weeks                        │
  ├──────────────────────────┼────────────────────┼──────────────────────────────┤
  │ TIR + better model (72B) │ 20–35              │ Weeks + compute              │
  ├──────────────────────────┼────────────────────┼──────────────────────────────┤
  │ TIR + fine-tuning        │ 30–40+             │ Months + significant compute │
  └──────────────────────────┴────────────────────┴──────────────────────────────┘

  The learning curve from 0 → working TIR submission is enormous. That's the real goal here — understanding
   how modern math AI systems are built from the ground up.

   ---                                                                                                      
  Step-by-Step: Running Notebook 01 on Kaggle                                                              
                                                                                                           
  Step 1 — Create the notebook correctly                                                                   
                                                                                                           
  From the screenshot you're in a blank Kaggle notebook. You need to upload 01_cot_baseline.ipynb instead
  of starting blank.

  1. Click File → Import Notebook
  2. Upload /Users/pulin05/kaggling/kaggling/Competitions/AIMO/notebooks/01_cot_baseline.ipynb

  ---
  Step 2 — Enable GPU

  This is critical. Without a GPU, the model will take hours.

  1. Click Settings (top menu) → Session Options (or the gear icon on the right panel)
  2. Set Accelerator → GPU T4 x2
  3. Click Save

  You'll see the session restart. The session indicator at the top changes from grey to green.

  ---
  Step 3 — Enable Internet

  The notebook downloads model weights from HuggingFace. You need internet ON for development notebooks (it
   gets turned off for final submission reruns).

  1. Click Settings → Session Options
  2. Toggle Internet → ON
  3. Click Save

  ---
  Step 4 — Add the Competition Dataset

  From the screenshot your right panel already shows the competition attached. Confirm it:

  1. In the right panel under Input, you should see "AI Mathematical Olympiad - Progress Prize..." listed
  2. If not: click + Add Input → search "ai-mathematical-olympiad-progress-prize-3" → click +

  This mounts the competition files at /kaggle/input/ai-mathematical-olympiad-progress-prize-3/.

  ---
  Step 5 — Fix the file path in the notebook

  The notebook currently reads:
  REFERENCE_CSV = '../Dataset/reference.csv'   # local path
  On Kaggle, the file is at a different location. You need to change one line before running.

  In the problems loading cell (the one with pd.read_csv), change:
  # Change this:
  REFERENCE_CSV = '../Dataset/reference.csv'

  # To this:
  REFERENCE_CSV = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv'

  ---
  Step 6 — Run All

  Click Run All (the ▶▶ button in the top toolbar).

  Expected sequence:
  - Cell 1 (!pip install): ~2 minutes — installs transformers, bitsandbytes, accelerate
  - Model loading cell: ~3–5 minutes — downloads ~4GB of weights from HuggingFace, quantizes to 4-bit
  - Each experiment (3 loops × 10 problems): ~10–15 min each

  Total: ~45–60 minutes.

  ---
  Why the Uninstall-Then-Reinstall Pattern?

  This is the core of the utility notebook you linked. Here's the full explanation:

  The problem: Ka   ggle's pre-installed packages are frozen

  Kaggle notebooks run inside a Docker container that has hundreds of packages pre-installed at fixed
  versions. These versions are often 6–12 months old. For example, Kaggle might have:

  transformers==4.38.0    ← pre-installed (old)
  bitsandbytes==0.41.0    ← pre-installed (old)

  But the latest Qwen2.5-Math models require:
  transformers>=4.45.0    ← needs new API
  bitsandbytes>=0.43.0    ← needs new CUDA kernels

  Simply doing pip install transformers does nothing if transformers is already installed — pip sees it's
  present and skips it. pip install --upgrade sometimes fails because the old version's files conflict with
   the new version's files mid-install.

  The solution: uninstall first, then install clean

  # Step 1: Remove the old version completely
  !pip uninstall -y transformers bitsandbytes

  # Step 2: Install the exact version you need
  !pip install transformers==4.47.0 bitsandbytes==0.44.0

  This guarantees no leftover files from the old version interfere with the new one.

  Why the AIMO utility notebook takes it further

  The competition submission has internet disabled during the official rerun (when your score is computed).
   So competitors pre-download everything:

  Utility Notebook 1 (run once, with internet ON):
    → pip install packages into /kaggle/working/
    → Save as a Kaggle Dataset

  Your submission notebook (internet OFF):
    → !pip install --no-index --find-links=/kaggle/input/my-packages/ transformers
    → Loads packages from your pre-saved dataset

  This two-notebook pattern is the standard AIMO workflow. You're not there yet — it's only needed for the
  final submission. For development (notebooks 01–04), just run with internet ON and the simple !pip
  install -q transformers bitsandbytes accelerate is enough.

  ---
  Quick Reference Card

  For development (now):               For competition submission (later):
  ─────────────────────────            ──────────────────────────────────
  Internet: ON                         Internet: OFF
  GPU: T4 x2                           GPU: T4 x2
  pip install directly                 Load packages from pre-saved dataset
  Path: /kaggle/input/aimo.../         Path: /kaggle/input/aimo.../
  Model: downloaded live               Model: pre-saved as Kaggle dataset

  Sources:
  - HuggingFace bitsandbytes quantization docs
  - Using transformers on Kaggle — HuggingFace Forums