# Topic 3: Neural Network Processing, Backward Propagation

## 1. Quick Recap: Forward Propagation

Last session, we saw that data flows **forward** through the network:

```
Layer 1: Z1 = W1 × Input + b1, then A1 = activation(Z1)
Layer 2: Z2 = W2 × A1 + b2, then A2 = activation(Z2)
Output:  Final = W3 × A2 + b3
```

Forward propagation answers: **"What does the network predict?"**

Today we answer the other half of the question: **"How does the network learn from its mistake?"** That's **backward propagation** (backpropagation).

---

## 2. Why Do We Need Backpropagation?

A network can easily have thousands (or millions) of weights and biases. After a forward pass, we get a **loss** — a single number telling us how wrong we were.

The problem: **which weights caused the error, and by how much?** We can't just guess-and-check millions of parameters one at a time — that would take forever.

**Backpropagation** is an efficient algorithm that computes the exact contribution of *every* weight and bias to the total error, all in a single backward sweep through the network.

---

## 3. The Chain Rule (Our Main Tool)

Backpropagation is really just the **chain rule** from calculus, applied layer by layer.

**The idea:** If `y` depends on `u`, and `u` depends on `x`, then:

```
dy/dx = dy/du × du/dx
```

**Simple example:**

- Let `u = 2x` and `y = u²`
- `dy/du = 2u` and `du/dx = 2`
- So `dy/dx = 2u × 2 = 4u = 8x`

A neural network is just a long chain of these small functions stacked together (multiply → add → activate → multiply → add → activate → ... → loss). The chain rule lets us compute how a change in *any* early weight affects the *final* loss, by multiplying together all the little derivatives in between.

---

## 4. The Computational Graph

It helps to think of the network as a **graph of operations**:

```
Input → [× W1, + b1] → Z1 → [activation] → A1 → [× W2, + b2] → Z2 → [activation] → A2 → [Loss]
```

- **Forward pass**: walk the graph left → right, computing values.
- **Backward pass**: walk the graph right → left, computing gradients (using the chain rule at every step).

**Analogy — "Playing the Blame Game":**

Imagine the loss is a manager who's unhappy with the final result. The manager doesn't blame the input directly — they blame the *last* person in the chain (the output layer), who then blames *their* supplier (the last hidden layer), who blames *their* supplier, and so on, all the way back to the input. Each person passes back "how much are you to blame," scaled by how much influence they had.

---

## 5. Backpropagation Through a Single Neuron

Let's trace it through one neuron:

```
z = w × x + b
a = activation(z)
Loss = (a - target)²
```

**Step 1 — How much does the loss change with the output `a`?**

```
dLoss/da = 2 × (a - target)
```

**Step 2 — How much does `a` change with `z`?** (derivative of the activation function)

```
da/dz = activation'(z)
```

**Step 3 — How much does `z` change with the weight `w`?**

```
dz/dw = x
```

**Chain them together:**

```
dLoss/dw = dLoss/da × da/dz × dz/dw
```

This single value — `dLoss/dw` — tells us exactly how to nudge `w` to reduce the loss (this is the **gradient** used in gradient descent from last session).

The same logic gives us `dLoss/db` (using `dz/db = 1`) and `dLoss/dx` (used to keep propagating the blame backward into earlier layers).

---

## 6. Backpropagation Through Multiple Layers

For a 2-layer network:

```
z1 = W1 × x + b1        a1 = activation(z1)
z2 = W2 × a1 + b2        a2 = activation(z2)
Loss = (a2 - target)²
```

The backward pass computes gradients **in reverse order**, reusing work from later layers:

```
dLoss/da2 = 2 × (a2 - target)
dLoss/dz2 = dLoss/da2 × activation'(z2)
dLoss/dW2 = dLoss/dz2 × a1
dLoss/db2 = dLoss/dz2

dLoss/da1 = dLoss/dz2 × W2              ← "blame" passed back to layer 1
dLoss/dz1 = dLoss/da1 × activation'(z1)
dLoss/dW1 = dLoss/dz1 × x
dLoss/db1 = dLoss/dz1
```

**Key insight:** `dLoss/dz2` (computed for the output layer) is *reused* to compute the gradient for layer 1. This reuse is exactly why backpropagation is efficient — each intermediate gradient is computed once and shared, instead of recomputing everything from scratch for every single weight.

---

## 7. Derivatives of Common Activation Functions

To run the chain rule, we need the *derivative* of whatever activation function we used:

| Activation | Function | Derivative |
| :--- | :--- | :--- |
| **ReLU** | `max(0, x)` | `1` if `x > 0`, else `0` |
| **Sigmoid** | `1 / (1 + e⁻ˣ)` | `sigmoid(x) × (1 - sigmoid(x))` |
| **Tanh** | `tanh(x)` | `1 - tanh(x)²` |

**Why this matters:** Sigmoid and Tanh derivatives max out at 0.25 and 1.0 respectively, and shrink toward **0** at the extremes. When many such small numbers get multiplied together across many layers, the gradient can shrink to almost nothing by the time it reaches the earliest layers — a preview of the **vanishing gradient problem** we'll cover in the next session.

---

## 8. The Full Backpropagation Algorithm

```
1. FORWARD PASS: Compute z and a for every layer, then the loss.
2. START AT THE OUTPUT: Compute dLoss/da for the final layer.
3. FOR EACH LAYER, GOING BACKWARD:
       a. Multiply by the activation derivative → get dLoss/dz
       b. Compute dLoss/dW and dLoss/db for this layer (using this layer's gradient)
       c. Compute dLoss/da for the PREVIOUS layer (to keep propagating backward)
4. UPDATE ALL WEIGHTS using gradient descent:
       W = W - learning_rate × dLoss/dW
       b = b - learning_rate × dLoss/db
```

This is the missing piece from last session's training loop — **"BACKWARD PASS: Calculate how to adjust weights"** is exactly backpropagation.

---

## 9. What We Learned Today

| Concept | What It Does |
| :--- | :--- |
| **Backpropagation** | Efficiently computes the gradient of the loss with respect to every weight |
| **Chain Rule** | The mathematical tool that lets gradients flow backward through layers |
| **Computational Graph** | A picture of forward computation that we traverse in reverse for gradients |
| **Gradient Reuse** | Later-layer gradients are reused to compute earlier-layer gradients (efficiency) |
| **Activation Derivatives** | Needed at every layer to convert `dLoss/da` into `dLoss/dz` |

---

## 10. What's Next?

In the next session, we'll learn about:

- **Train, Test & Validation Sets**: How to properly evaluate a network
- **Vanishing & Exploding Gradients**: What happens when backprop's chain of multiplications goes wrong
- **Dropout & Regularization**: Techniques to prevent overfitting

---

## Key Takeaway

> Backpropagation is the chain rule applied systematically, layer by layer, from the loss back to the inputs. It computes the exact gradient for every weight in one efficient backward sweep, which is then handed to gradient descent to actually update the weights. Together, **forward propagation + backpropagation + gradient descent** form the complete training loop of a neural network.
