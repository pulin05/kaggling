# Recurrent Neural Networks (RNNs) — From Scratch

> Goal: Build intuition first, then math, then code. Every concept is grounded in *why* before *how*.

---

## Table of Contents

1. [Why Do We Need RNNs?](#1-why-do-we-need-rnns)
2. [The Core Idea: Memory via Recurrence](#2-the-core-idea-memory-via-recurrence)
3. [RNN Architecture — Step by Step](#3-rnn-architecture--step-by-step)
4. [Unrolling Through Time](#4-unrolling-through-time)
5. [Forward Pass — The Math](#5-forward-pass--the-math)
6. [Backpropagation Through Time (BPTT)](#6-backpropagation-through-time-bptt)
7. [The Vanishing Gradient Problem](#7-the-vanishing-gradient-problem)
8. [The Exploding Gradient Problem](#8-the-exploding-gradient-problem)
9. [Solutions Overview](#9-solutions-overview)
10. [Types of RNN Architectures](#10-types-of-rnn-architectures)
11. [Mental Model Summary](#11-mental-model-summary)

---

## 1. Why Do We Need RNNs?

### The problem with vanilla feedforward networks

A standard neural network (MLP) takes a **fixed-size input** and produces a **fixed-size output**. It has **no memory** — every input is treated independently.

This breaks for **sequential data**, where the meaning of each element depends on what came before:

```
"The bank can guarantee deposits will eventually cover future tuition costs"
"He sat on the bank of the river"
```

The word *bank* means something completely different depending on context. An MLP processing one word at a time has no way to track that context.

**Examples of sequential data:**
| Domain | Sequence | Dependency |
|--------|----------|------------|
| NLP | Words in a sentence | Grammar, coreference |
| Time series | Stock prices | Trends, seasonality |
| Audio | Sound waves | Phonemes across time |
| Video | Frames | Motion, causality |
| Biology | DNA sequences | Gene expression |

### What we actually need

A model that:
1. Can handle **variable-length** inputs/outputs
2. **Shares parameters** across time steps (don't re-learn "what a verb is" for each position)
3. Maintains a **running summary** of what it has seen so far

This is exactly what RNNs provide.

---

## 2. The Core Idea: Memory via Recurrence

### The intuition

Think of reading a book. At any point, you carry a mental summary of what you've read so far. When you read the next sentence, you:
1. Update your summary with the new information
2. Use that updated summary to understand the sentence

An RNN does exactly this. It maintains a **hidden state** `h_t` — a vector that acts as a compressed memory of everything seen up to time `t`.

### The recurrence relation

```
h_t = f(h_{t-1}, x_t)
```

- `x_t` = current input at time step t
- `h_{t-1}` = hidden state from previous time step (the "memory")
- `h_t` = new hidden state (updated memory)
- `f` = a learned function (the RNN cell)

**Key insight:** The same function `f` (same weights) is applied at every time step. This is **parameter sharing** — the network learns one general rule for updating memory, not a separate rule for each position.

---

## 3. RNN Architecture — Step by Step

### Single time step view

```
         y_t  (output, optional)
          ↑
     [Output layer]
          ↑
    h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
          ↑              ↑
       (memory)     (new input)

    h_{t-1} ──────────────────┐
                              │
    x_t ──────────────────────┤
                              ↓
                         [RNN Cell]──→ h_t ──→ (next step)
```

### The three weight matrices

| Matrix | Shape | Role |
|--------|-------|------|
| `W_x` | (hidden_size × input_size) | How input affects state |
| `W_h` | (hidden_size × hidden_size) | How past state affects new state |
| `W_y` | (output_size × hidden_size) | How state maps to output |

**Critically:** `W_x`, `W_h`, `W_y` are **shared across all time steps**. The whole sequence is processed with the same parameters.

### Parameter count example

If input_size=10, hidden_size=50, output_size=5:
- `W_x`: 50×10 = 500 params
- `W_h`: 50×50 = 2,500 params (this is the memory connection!)
- `W_y`: 5×50 = 250 params
- Biases: 50 + 5 = 55 params
- **Total: ~3,305 params** regardless of sequence length

---

## 4. Unrolling Through Time

An RNN processing a sequence of length T can be "unrolled" into a deep feedforward network with T layers, where all layers **share weights**.

```
x_1      x_2      x_3      x_4
 ↓        ↓        ↓        ↓
[RNN] → [RNN] → [RNN] → [RNN]
 ↓        ↓        ↓        ↓
h_1      h_2      h_3      h_4
         ↓                  ↓
        y_2               y_4   ← (outputs wherever needed)
```

This unrolled view is crucial for understanding:
- **Forward pass**: flows left to right
- **Backprop**: flows right to left (through time)
- **The vanishing gradient problem**: gradients must travel the full length of this chain

---

## 5. Forward Pass — The Math

### Notation

- `T` = sequence length
- `h_0` = initial hidden state (usually zeros)
- `σ` = activation function (tanh is standard for vanilla RNN)

### Step-by-step computation

**At each time step t = 1, 2, ..., T:**

```
Step 1: Compute pre-activation
    a_t = W_h · h_{t-1} + W_x · x_t + b_h

Step 2: Apply activation (squash into [-1, 1])
    h_t = tanh(a_t)

Step 3: Compute output (if needed at this step)
    o_t = W_y · h_t + b_y
    y_t = softmax(o_t)  ← for classification
```

### Why tanh (not ReLU)?

- tanh squashes values to `[-1, 1]` — prevents runaway state values
- Centered at 0 — better gradient flow than sigmoid
- BUT: its gradient saturates near ±1 (this causes problems — see Section 7)

### Loss computation

For sequence-to-sequence tasks, loss is summed over all time steps:

```
L = Σ_{t=1}^{T} L_t(y_t, ŷ_t)
```

---

## 6. Backpropagation Through Time (BPTT)

### What is BPTT?

Standard backprop on the **unrolled** RNN. Gradients flow backward through both:
1. The output at each step
2. The hidden state chain from right to left

### The chain rule through time

To update `W_h`, we need `∂L/∂W_h`. Using the chain rule:

```
∂L/∂W_h = Σ_{t=1}^{T} ∂L_t/∂W_h
```

For a single loss term `L_t`, the gradient must flow back through all previous hidden states:

```
∂L_t/∂W_h = Σ_{k=1}^{t} (∂L_t/∂h_t · ∏_{j=k+1}^{t} ∂h_j/∂h_{j-1}) · ∂h_k/∂W_h
```

The key term is the **product of Jacobians**:

```
∏_{j=k+1}^{t} ∂h_j/∂h_{j-1}
```

This product is what causes the vanishing (or exploding) gradient problem.

### Computing ∂h_t/∂h_{t-1}

Since `h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)`:

```
∂h_t/∂h_{t-1} = diag(1 - h_t²) · W_h
```

Where `diag(1 - h_t²)` is the Jacobian of tanh (its derivative is `1 - tanh²`).

---

## 7. The Vanishing Gradient Problem

### What happens over long sequences

When you multiply the Jacobian `∂h_j/∂h_{j-1}` repeatedly (T times), two things can happen:

```
If ||∂h_j/∂h_{j-1}|| < 1  →  gradient shrinks exponentially  →  VANISHING
If ||∂h_j/∂h_{j-1}|| > 1  →  gradient grows exponentially   →  EXPLODING
```

For tanh, the derivative `(1 - tanh²(x))` is in `[0, 1]`. So:

```
||∂h_t/∂h_k|| ≤ (λ_max · 1)^(t-k)
```

Where `λ_max` is the largest singular value of `W_h`. If this product is < 1, gradients vanish.

### Concrete example

Suppose the gradient shrinks by factor 0.5 at each step.
- After 10 steps: `0.5^10 ≈ 0.001` — already tiny
- After 50 steps: `0.5^50 ≈ 10^{-15}` — effectively zero

**The gradient from step 50 is completely invisible at step 1.** The network cannot learn that an event early in the sequence matters for a later outcome.

### Why this matters — the long-range dependency problem

```
"The cat that sat on the mat, which was old and dusty, ______ hungry."
```

To predict the verb must agree with "cat" (singular), but "cat" is 10+ tokens away. A vanilla RNN trained with BPTT can't propagate the gradient from the final verb back to where "cat" appeared — the gradient vanishes before it gets there.

### What the network actually learns

With vanishing gradients, the hidden state `h_t` effectively only "remembers" the last ~5-10 time steps. Everything earlier is forgotten because there's no gradient signal to reinforce those connections.

### Visualizing gradient flow

```
Steps:  t=1    t=2    t=3    t=4    t=5   ...  t=50
        [RNN]→[RNN]→[RNN]→[RNN]→[RNN]→ ... →[RNN]→ Loss

Grad:    ←  ←  ←  ←  ←  ←  ← ...

         by t=1, gradient is ~0
         (early weights receive no learning signal)
```

### Root cause — three compounding factors

1. **Repeated multiplication**: Same `W_h` applied T times = eigenvalue amplification
2. **Saturating activations**: tanh derivative is ≤ 1, and near 0 at saturation
3. **Chain rule depth**: Each additional time step adds one more multiplication

---

## 8. The Exploding Gradient Problem

Less common but also dangerous. When `||W_h|| > 1`, gradients grow exponentially.

**Symptoms:**
- Loss becomes NaN
- Weights blow up to ±inf
- Training immediately diverges

**Fix:** Gradient clipping — if `||∇||  > threshold`, rescale it:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This is a simple, widely-used patch. Vanishing gradients have no such easy fix.

---

## 9. Solutions Overview

### 9.1 LSTM (Long Short-Term Memory)

The dominant solution. Adds a **cell state** `c_t` — a "highway" for gradients to flow through time — controlled by learned gates.

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   ← what to erase
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   ← what to write
Cell update:  c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
New cell:     c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t       ← additive update!
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

**Why it fixes vanishing gradients:** The cell state update is **additive** (`c_t = f_t ⊙ c_{t-1} + ...`). During backprop, gradients flow through addition — no repeated multiplication through tanh. The forget gate can learn to be ≈1, creating an identity path for gradients.

### 9.2 GRU (Gated Recurrent Unit)

Simplified LSTM with 2 gates instead of 3. Slightly less powerful but faster:

```
Reset gate:  r_t = σ(W_r · [h_{t-1}, x_t])
Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
Candidate:   h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
New state:   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 9.3 Gradient Clipping

For exploding gradients only. Simple but essential in practice.

### 9.4 Truncated BPTT

Only backprop through the last K steps (not the full sequence). Limits the depth of the gradient chain. Used in practice for very long sequences.

### 9.5 Attention Mechanisms → Transformers

The modern solution. Instead of a sequential chain, **every token attends directly to every other token** — O(1) path length for any dependency. This eliminates the vanishing gradient problem structurally. Transformers have largely replaced RNNs for NLP.

---

## 10. Types of RNN Architectures

### By input/output structure

```
One-to-One        One-to-Many       Many-to-One       Many-to-Many
  x → y           x → y1,y2,y3    x1,x2,x3 → y     x1,x2 → y1,y2
  (MLP)          (image caption)   (sentiment)        (translation)
```

### Bidirectional RNN

Runs two RNNs: one forward (past → future) and one backward (future → past). Concatenates both hidden states.

```
→ [RNN] → [RNN] → [RNN] →    (forward)
← [RNN] ← [RNN] ← [RNN] ←    (backward)
     ↓        ↓        ↓
   [h_f, h_b] at each step
```

**When to use:** When full sequence is available at inference time (NER, classification). Cannot be used for generation (you don't have future tokens).

### Deep (Stacked) RNN

Multiple RNN layers. Output of layer 1 becomes input to layer 2.

```
Layer 2: [RNN] → [RNN] → [RNN]
              ↑        ↑        ↑
Layer 1: [RNN] → [RNN] → [RNN]
              ↑        ↑        ↑
         x_1      x_2      x_3
```

---

## 11. Mental Model Summary

| Concept | Mental Model |
|---------|-------------|
| Hidden state `h_t` | A running notepad — compressed summary of all past inputs |
| `W_h` | How much the notepad influences itself each step |
| `W_x` | How much the new input updates the notepad |
| Unrolling | Treating the RNN as a very deep network where all layers share weights |
| BPTT | Backprop through that deep unrolled network |
| Vanishing gradient | The gradient "fades out" as it travels backwards — early steps get no signal |
| LSTM cell state | A protected highway for long-range information to travel without decay |
| Attention | Bypassing the chain entirely — every token sees every other token directly |

### The progression of ideas

```
MLP (no memory)
    ↓  Problem: can't handle sequences
RNN (hidden state as memory)
    ↓  Problem: vanishing gradients, can't learn long-range deps
LSTM/GRU (gated memory highways)
    ↓  Problem: still sequential, slow to train
Transformer (attention = direct connections everywhere)
    ↓  Current state of the art for most sequence tasks
```

---

*Next steps: implement a vanilla RNN in NumPy from scratch, then build an LSTM and compare gradient norms over sequence length.*
