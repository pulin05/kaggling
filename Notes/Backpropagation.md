# Backpropagation

## History

### The Credit Assignment Problem (1960s)
The fundamental challenge: in a network of connected units, when the output is wrong, *which* unit is responsible? How do you assign "credit" or "blame" to each weight? This was the central unsolved problem in neural network research for nearly two decades.

### Early Attempts
- **Rosenblatt's Perceptron (1958)** had a learning rule, but it only worked for *single-layer* networks. There was no principled way to train hidden layers.
- **Minsky & Papert's "Perceptrons" (1969)** proved single-layer networks couldn't solve XOR — this killed funding and interest for a decade. The implicit claim: multilayer networks can't be trained either.

### The Rediscovery (1970–1986)
Backprop was independently derived multiple times before it became famous:
- **Seppo Linnainmaa (1970)** — described reverse-mode automatic differentiation in his master's thesis. This is mathematically identical to backprop.
- **Paul Werbos (1974)** — derived backprop through neural networks in his Harvard PhD thesis. Almost entirely ignored.
- **Parker (1985)** — rediscovered it independently.
- **Rumelhart, Hinton & Williams (1986)** — published "Learning representations by back-propagating errors" in *Nature*. This is the paper that made backprop famous. They showed it could learn useful internal representations (features) in hidden layers — directly refuting the Minsky & Papert pessimism.

### Why 1986 Succeeded Where 1974 Failed
Werbos had the math right. What changed:
1. **Computing power** — machines were fast enough to actually run the experiments
2. **The framing** — Rumelhart et al. emphasized *learned representations*, not just function approximation
3. **The timing** — the field was ready; connectionism was resurging

### Post-1986 Development
- **1989** — Universal Approximation Theorem proved (Hornik et al.): a 2-layer network with enough hidden units can approximate any continuous function
- **1990s** — Vanishing gradient problem discovered (Hochreiter 1991, Bengio et al. 1994): backprop fails in deep networks because gradients shrink exponentially
- **1997** — LSTM (Hochreiter & Schmidhuber) designed specifically to fight vanishing gradients
- **2006** — Hinton's deep belief nets showed deep networks *could* be trained (with pre-training tricks)
- **2012** — AlexNet. GPUs + ReLU + dropout + more data. Deep learning era begins.

---

## Core Concepts

### What Backpropagation Actually Is
Backprop is an efficient algorithm for computing **the gradient of a loss function with respect to every weight** in a neural network. It's not a learning algorithm itself — it just computes gradients. Gradient descent (or Adam, etc.) uses those gradients to update weights.

The key insight: **the chain rule of calculus, applied systematically in reverse.**

---

### Forward Pass
Given input **x**, compute the network's prediction **ŷ** layer by layer:

```
x → [W1, b1] → z1 = W1·x + b1 → a1 = f(z1) → [W2, b2] → z2 → a2 = ŷ
```

- **z** = pre-activation (linear combination)
- **a** = post-activation (after applying activation function)
- Every intermediate value is *cached* — you'll need them in the backward pass

At the end, compute the loss: `L = loss(ŷ, y)`

---

### The Chain Rule
If `L` depends on `a`, and `a` depends on `z`, and `z` depends on `w`, then:

```
dL/dw = (dL/da) · (da/dz) · (dz/dw)
```

Backprop is just this, applied to a computation graph with thousands of weights. The "backward" direction refers to computing these derivatives from output back to input.

---

### Backward Pass

**Step 1 — Output gradient:**
```
dL/dŷ   ← derivative of loss w.r.t. prediction (e.g., ŷ - y for MSE)
```

**Step 2 — Propagate through each layer, in reverse:**

For a layer with weights W, bias b, pre-activation z = W·a_prev + b, and activation a = f(z):

```
δ = dL/dz = (dL/da) · f'(z)       ← "delta" or "error signal" for this layer
dL/dW = δ · a_prev^T              ← gradient w.r.t. weights
dL/db = δ                         ← gradient w.r.t. bias
dL/da_prev = W^T · δ              ← pass error signal to previous layer
```

Repeat for each layer until you reach the input.

**Step 3 — Update weights:**
```
W ← W - lr · dL/dW
b ← b - lr · dL/db
```

---

### Why It's Efficient: Dynamic Programming
A naive approach would compute the gradient for each weight independently — O(weights²) operations. Backprop avoids this by reusing intermediate computations.

The gradient at layer *k* depends on the gradient at layer *k+1* (already computed). So you compute each gradient exactly once, in one backward sweep: **O(weights)** — the same cost as the forward pass.

This is reverse-mode automatic differentiation. (Forward-mode AD exists too but is efficient only when #inputs << #outputs — the opposite of neural nets.)

---

### Activation Functions and Their Derivatives

| Activation | f(z) | f'(z) | Notes |
|---|---|---|---|
| Sigmoid | 1/(1+e^-z) | f(z)·(1-f(z)) | Saturates → vanishing gradients |
| Tanh | (e^z - e^-z)/(e^z + e^-z) | 1 - f(z)² | Zero-centered; still saturates |
| ReLU | max(0, z) | 1 if z>0, else 0 | No saturation for z>0; dying ReLU problem |
| Leaky ReLU | max(αz, z) | 1 if z>0, else α | Fixes dying ReLU |
| Softmax | e^zi / Σe^zj | (complex; often paired with cross-entropy) | Output layer for classification |

The derivative of the activation function is the key term in backprop — it controls how much gradient flows through each neuron.

---

### The Vanishing Gradient Problem
In a deep network with sigmoid activations, the gradient for layer 1 is:

```
dL/dW1 = dL/dWn · f'(z_{n-1}) · W_{n-1} · f'(z_{n-2}) · W_{n-2} · ... · f'(z_1)
```

Sigmoid's derivative is at most 0.25. Multiply 10 of these together: 0.25^10 ≈ 0.000001. The gradient vanishes — early layers learn nothing.

**Solutions:**
- **ReLU** — derivative is 1 for positive activations, doesn't shrink
- **Batch Normalization** — keeps activations in a healthy range
- **Residual connections (ResNets)** — create gradient highways that skip layers
- **Better initialization** (Xavier/He) — prevents activations from saturating at initialization

---

### Computational Graphs
Modern deep learning frameworks (PyTorch, JAX) implement backprop via **automatic differentiation** on a DAG (directed acyclic graph):

- Each node is an operation (add, multiply, matmul, ReLU...)
- Each edge carries a tensor
- Forward pass: evaluate the graph, cache intermediate values
- Backward pass: traverse the graph in reverse, apply local gradients via chain rule

PyTorch builds this graph dynamically (define-by-run). JAX traces through Python functions. Either way, you write forward pass code; the framework handles the backward pass automatically.

---

### Numerical Gradient Check
Before trusting your backprop implementation, verify it:

```
dL/dw ≈ [L(w + ε) - L(w - ε)] / (2ε)     for small ε (e.g., 1e-5)
```

Compare this to your analytical gradient. If they match to ~5 significant figures, your backprop is correct. This is the standard debugging technique.

---

## Summary: The Big Picture

```
Forward pass:  x → network → ŷ → L        (compute loss, cache intermediates)
Backward pass: L → dL/dŷ → dL/dW_n → ... → dL/dW_1   (chain rule, reverse)
Update:        W ← W - lr · dL/dW          (gradient descent)
```

Backprop's contribution is making the backward pass computationally feasible — O(n) instead of O(n²) — by recognizing that gradients can be composed and shared across the graph. Everything else (architectures, optimizers, regularization) builds on top of this foundation.
