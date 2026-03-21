# Nemotron-3-Nano-30B Architecture Explained

```
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
├── Total params:  30B
├── Active params: 3.5B per token  ← only a fraction runs per forward pass
└── 52 layers total:
    ├── 23 Mamba-2 layers    → state-space model (SSM), not attention
    ├── 23 MoE layers        → 128 experts, only 6 active per token
    └──  6 Attention layers  → GQA with 2 groups (memory-efficient)
```

---

## The Core Job of Any Language Model

Given a sequence of tokens (words), predict the next token. To do that well, the model needs to **remember context** — what came before.

The question is: *how do you store and use that memory efficiently?*

---

## Layer Type 1: The Attention Layer (the classic)

Standard Transformers use **attention**: every token looks at every other token and decides "how much should I pay attention to you?"

```
Token 5 ("cat") → looks back at tokens 1,2,3,4 → weights how relevant each is → builds understanding
```

**Problem:** This is O(n²) in memory and compute — as the sequence gets longer, it gets *quadratically* more expensive. At 100K tokens, it's brutal.

**GQA (Grouped Query Attention)** is just an optimization: instead of each "query" having its own "key+value" heads, they *share* them in groups. This model uses 2 groups — meaning 2× less memory than standard attention. Same idea, less RAM.

This model only uses **6 attention layers** out of 52. Attention is powerful but expensive — so use it sparingly, only where global context is critical.

### What is "global context"?

Consider this sentence:

> *"The trophy didn't fit in the suitcase because **it** was too big."*

What does **it** refer to — the trophy or the suitcase?

To answer that, the token "it" needs to look back at both "trophy" and "suitcase" and reason about which one is too big. That's global context — understanding a token by comparing it to *specific other tokens far away* in the sequence.

Attention does this perfectly:

```
"it" → scores every past token → trophy gets high score → "it" = trophy
```

### Why can't Mamba do this?

Mamba compresses everything into a fixed-size state as it goes. Think of it like taking notes while reading:

```
Read "The trophy"     → state = {trophy: large object}
Read "didn't fit"     → state = {trophy: large, fit: failed}
Read "in suitcase"    → state = {trophy: large, suitcase: container, fit: failed}
Read "because it..."  → state = {...summary so far...}
```

By the time you reach "it", the detailed information about "trophy" has been compressed and partially overwritten by everything that came after. You can't go back and look at the original "trophy" token directly.

Mamba is great at tracking trends and patterns through a sequence, but bad at precise point lookups — "give me exactly what token 7 said."

### So why only 6 attention layers?

Most of language understanding doesn't need precise long-range lookups. Reading a sentence word by word, building up meaning incrementally — that's Mamba's job, and it's cheap.

But occasionally the model hits a moment where it genuinely needs to ask *"wait, what exactly did that earlier token say?"* — that's when attention earns its cost.

The designers found empirically: 6 attention layers scattered across 52 is enough to handle those precise-recall moments, while the 23 Mamba layers handle the rest cheaply.

It's the same reason you don't call a meeting for every decision — you handle most things yourself (Mamba), and only convene the full team (Attention) when you actually need everyone's input.

---

## Layer Type 2: Mamba-2 Layers (the new kid)

Mamba is a **State Space Model (SSM)**. Instead of looking at all past tokens, it maintains a **compressed hidden state** — like a fixed-size summary of everything seen so far.

Think of it like working memory vs. a filing cabinet:

```
Attention = filing cabinet (can look up any past token, but the cabinet grows forever)
Mamba     = working memory (fixed size, continuously updated as new tokens arrive)
```

How Mamba updates its state:

```
new_state = A × old_state + B × new_token
output    = C × new_state
```

`A`, `B`, `C` are learned matrices. The model learns *what to keep, what to forget, what to output*.

**Why is this fast?** The state size is **fixed** regardless of sequence length. Processing 1M tokens costs the same as 1K tokens. It's O(n) not O(n²).

**Mamba-2** specifically restructures these matrices to run efficiently on modern GPUs (uses structured state spaces that map well to matrix multiplications).

This model uses **23 Mamba-2 layers** — the workhorse for sequence compression.

---

## Layer Type 3: MoE Layers (the efficiency trick)

**Mixture of Experts** answers the question: *how do you make a model with 30B parameters but only use 3.5B at a time?*

Each MoE layer has 128 "expert" sub-networks (small FFN modules). A **router** network looks at the current token and picks the 6 most relevant experts:

```
Token arrives → Router scores all 128 experts → Top-6 activated → Token processed through those 6 only
```

So instead of running 30B parameters per token, you only run ~3.5B. The model has *capacity* of 30B (it can specialize), but *cost* of 3.5B.

Different tokens route to different experts — "math tokens" might activate expert #17 and #43, while "code tokens" activate #2 and #91. The model self-organizes this during training.

This model uses **23 MoE layers** — the 23 Mamba layers each have a corresponding MoE feedforward block.

---

## How the 52 Layers Stack Together

The full architecture interleaves these layer types:

```
Input tokens
    ↓
[Mamba-2 + MoE]  ← compress sequence, route through experts
[Mamba-2 + MoE]
[Mamba-2 + MoE]
    ...  (×23 pairs)
[Attention + MoE] ← occasionally do global context lookup
    ...  (×6 attention layers scattered in)
    ↓
Output logits → next token probability
```

The intuition for the hybrid:
- **Mamba layers** handle the bulk of sequence compression cheaply
- **Attention layers** are inserted strategically where the model needs to do precise, long-range lookups (can't be approximated by state compression)
- **MoE** on top of both ensures that at any layer, only the specialized subset of parameters fires

---

## Why This Design? (The Big Picture)

| Problem | Solution |
|---|---|
| Attention is O(n²) expensive | Use Mamba (O(n)) for most layers |
| Mamba can't do precise recall | Keep 6 attention layers for global lookups |
| 30B params too slow to run fully | MoE: activate only 3.5B per token |
| Need specialization at scale | 128 experts self-organize by token type |

The result: a model that has the **capacity** of a 30B model, the **inference cost** of a ~3.5B model, and handles **long sequences** better than a pure Transformer.

---

## Mental Model Summary

```
Mamba     = efficient conveyor belt — moves information forward cheaply
MoE       = specialist team — only the right experts show up for each token
Attention = occasional all-hands meeting — expensive but sometimes necessary
```

The hybrid uses each where it's strongest. That's the architecture.