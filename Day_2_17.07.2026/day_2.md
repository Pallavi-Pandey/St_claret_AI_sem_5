# Topic 2: Neural Network Overview

## 1. What is a Neural Network? (Quick Recap)

A neural network is like a team of tiny workers (neurons) organized in layers. Each worker receives numbers, does a simple calculation, and passes the result to the next layer.

**Think of it like this:**

- **Input Layer**: The receptionist — receives the raw data
- **Hidden Layers**: The workers — process and transform the data
- **Output Layer**: The manager — gives the final answer

---

## 2. How Does a Neuron Work?

Each neuron does just **two things**:

### Step 1: Multiply and Add

Each input gets multiplied by a **weight** (importance score), then we add a **bias** (threshold).

```
weighted_sum = (input1 × weight1) + (input2 × weight2) + bias
```

**Simple Example:**

- Input1 = 0.5, Weight1 = 0.3
- Input2 = 0.8, Weight2 = 0.7
- Bias = 0.1

```
weighted_sum = (0.5 × 0.3) + (0.8 × 0.7) + 0.1
             = 0.15 + 0.56 + 0.1
             = 0.81
```

### Step 2: Apply Activation Function

We pass the weighted sum through an **activation function** to decide what to pass forward.

**Why?** Without this, the network could only learn simple straight-line patterns. Activation functions let it learn complex patterns like recognizing faces or understanding language.

---

## 3. Forward Propagation (How Data Flows)

**Forward propagation** is just the process of data flowing from input to output, layer by layer.

**Think of it like an assembly line:**

1. Raw data enters the input layer
2. Each hidden layer processes it (multiply + add → activation)
3. The output layer gives the final prediction

**The Math (simplified):**

```
Layer 1: Z1 = W1 × Input + b1, then A1 = activation(Z1)
Layer 2: Z2 = W2 × A1 + b2, then A2 = activation(Z2)
Output:  Final = W3 × A2 + b3
```

---

## 4. Activation Functions (The Magic Sauce)

Activation functions decide how much of the signal to pass forward. Here are the main ones:

### ReLU (Most Popular)

```
If x > 0 → output x
If x ≤ 0 → output 0
```

**Why so popular?** Simple, fast, and works well in practice.

### Sigmoid

```
Squashes any number between 0 and 1
```

**Used for:** Binary classification (yes/no questions)

### Softmax

```
Converts a list of numbers into probabilities that add up to 1
```

**Used for:** Choosing between multiple options (like digits 0-9)

### Why Do We Need Non-Linearity?

If we only used linear functions (like y = 2x + 1), stacking layers would be pointless — multiple linear layers collapse into one simple line. Non-linear activation functions let the network learn complex, curved patterns.

---

## 5. Loss Functions (How Wrong Are We?)

The **loss function** measures how far off our predictions are from the correct answers.

### For Regression (Predicting Numbers)

**Mean Squared Error (MSE):**

```
Loss = average of (actual - predicted)²
```

- If we predict 4.5 but the answer is 5.0 → error = 0.5 → squared = 0.25
- Squaring makes big errors hurt more

### For Classification (Choosing Categories)

**Cross-Entropy Loss:**

- Heavily penalizes confident wrong answers
- Example: If we're 90% sure it's "cat" but it's actually "dog" → huge penalty

---

## 6. Gradient Descent (How the Network Learns)

**Gradient descent** is the algorithm that actually learns. Here's the idea:

### The Mountain Analogy 🏔️

Imagine you're on a mountain in thick fog, and you want to get to the lowest point:

1. **Feel the slope** under your feet (that's the gradient)
2. **Take a step** downhill (that's the update)
3. **Repeat** until you reach the bottom

### The Update Rule

```
new_weight = old_weight - learning_rate × gradient
```

- **Learning rate**: How big each step is (too big = overshoot, too small = takes forever)
- **Gradient**: Direction and steepness of the slope

**Simple Example:**

Say a weight `w = 0`, and the loss function is `Loss = (w - 2)² + 1` (its lowest point, the "bottom of the mountain," is at `w = 2`).

- **Gradient** at `w = 0`: `2 × (w - 2) = 2 × (0 - 2) = -4`
- With a **learning rate of 0.1**:

```
new_weight = old_weight - learning_rate × gradient
           = 0 - 0.1 × (-4)
           = 0 + 0.4
           = 0.4
```

- **Loss before the step:** `(0 - 2)² + 1 = 5.0`
- **Loss after the step:** `(0.4 - 2)² + 1 = 3.56`

One step already reduced the loss from 5.0 to 3.56! Repeating this update over and over keeps nudging `w` closer to 2, where the loss is smallest.

### Choosing the Learning Rate

The learning rate controls how big each step is, and getting it wrong causes two very different problems:

| Learning Rate | What Happens | Why |
| :--- | :--- | :--- |
| **Too small** (e.g. 0.001) | Learning is painfully slow | Tiny steps mean it takes forever to reach the bottom |
| **Just right** (e.g. 0.1–0.5) | Converges quickly and smoothly | Steps are big enough to make progress, small enough to settle down |
| **Too large** (e.g. 1.1+) | Overshoots and can diverge | Each step jumps past the minimum, and the loss gets *worse* over time |

### Types of Gradient Descent

The example above computed the gradient from a single weight for simplicity. In practice, the gradient is computed from *training examples* — and how many examples we use per update gives us three variants. First, a key distinction:

- **Epoch**: one complete pass through the *entire* training dataset
- **Update (a.k.a. "step" or "iteration")**: one single adjustment of the weights

An epoch can contain **one** update or **thousands** of updates — that's exactly what changes between the three variants below. Let's say we have a dataset of **1,000 training examples**.

#### 1. Batch Gradient Descent (BGD)

Uses the **entire dataset** to compute one gradient, then makes a single update.

```
1,000 examples ÷ 1,000 examples-per-update  =  1 update per epoch
```

- **Pros**: The gradient is averaged over every example, so it's very accurate — the loss curve is smooth, with no noise.
- **Cons**: You must process the whole dataset just to take one step. For a huge dataset, that's painfully slow, and the whole dataset must fit in memory at once.
- *(This is what our hands-on notebook uses — every update looks at all 100 data points at once, which is why its loss curve is so smooth.)*

#### 2. Stochastic Gradient Descent (SGD)

Uses just **one randomly chosen example** to compute the gradient and update the weights immediately.

```
1,000 examples ÷ 1 example-per-update  =  1,000 updates per epoch
```

- **Pros**: Starts improving immediately, needs almost no memory per step, and can escape small bumps in the loss landscape thanks to its randomness.
- **Cons**: Each gradient is estimated from a single data point, so it's noisy — the loss bounces around a lot instead of decreasing smoothly (though it still trends downward overall).

#### 3. Mini-Batch Gradient Descent

A middle ground: uses a small **batch** of examples (commonly 32–256) per update.

```
1,000 examples ÷ 100 examples-per-batch  =  10 updates per epoch
```

- **Pros**: Much smoother than SGD (averaging over 100 examples cancels out a lot of noise), while still far cheaper per update than full Batch GD. It's also the only variant that fully takes advantage of fast, vectorized/GPU computation.
- **Cons**: The batch size becomes a new hyperparameter to tune — too small behaves like noisy SGD, too large behaves like slow Batch GD.

**In practice:** almost every real-world deep learning model is trained with Mini-Batch Gradient Descent — that's exactly why every training framework (TensorFlow, PyTorch, etc.) asks you to set a `batch_size`.

| Type                    | How it Works               | Updates per Epoch (1,000 examples) | Pros         | Cons                    |
| ----------------------- | -------------------------- | ----------------------------------- | ------------ | ----------------------- |
| **Batch GD**      | Uses ALL data each time    | 1                                    | Stable       | Slow                    |
| **Stochastic GD** | Uses 1 sample at a time    | 1,000                                | Fast         | Noisy                   |
| **Mini-Batch GD** | Uses small groups (32–256) | ~4–31                                | Best of both | Need to pick batch size |

---

## 7. The Training Loop (Putting It All Together)

Everything from Sections 2–6 — the neuron's math, the loss function, and the gradient descent update rule — comes together into one repeating loop. This loop, run over and over, *is* how a network learns:

```
For each epoch (one complete pass through all data):
    For each batch of data:
        1. FORWARD PASS: Feed data through network → get prediction
        2. COMPUTE LOSS: Compare prediction to correct answer
        3. BACKWARD PASS: Calculate how to adjust weights
        4. UPDATE: Adjust weights using gradient descent
```

**Note:** the inner `For each batch` loop is where the choice from Section 6 (Batch / Stochastic / Mini-Batch GD) comes in. Our hands-on notebook uses **Batch GD**, so its "batch" is the *entire* dataset — meaning that inner loop only ever runs **once** per epoch. If we were using Mini-Batch GD instead, that inner loop would run several times per epoch, once per mini-batch, with steps 1–4 repeating for each one.

### Tracing One Epoch, Step by Step

Say we're partway through training, and the current weights give a loss of `0.05`.

1. **Forward pass** — the data flows through the network (Section 3) and produces a prediction, e.g. `0.83` for a target of `1.0`.
2. **Compute loss** — we compare `0.83` to `1.0` using a loss function (Section 5), e.g. MSE gives `(1.0 - 0.83)² = 0.0289`.
3. **Backward pass** — the network works out *how much each individual weight contributed* to that `0.0289` loss. (This step — **backpropagation** — is exactly what we'll learn to compute by hand next session; for now, think of it as "figuring out each weight's share of the blame.")
4. **Update** — every weight gets nudged using the update rule from Section 6: `new_weight = old_weight - learning_rate × gradient`.

Repeat this thousands of times (once per epoch, or more often if using mini-batches), and the loss keeps shrinking — that's a network learning.

### Key Terms

- **Epoch**: One complete pass through the entire training dataset.
- **Batch**: The group of samples used to compute *one* gradient/update. Its size is what separates Batch GD (batch = everything), Stochastic GD (batch = 1), and Mini-Batch GD (batch = a small group) — see Section 6.
- **Learning Rate**: The step size for each weight update (Section 6's "Choosing the Learning Rate").
- **Convergence**: The point where the loss stops meaningfully improving from one epoch to the next — e.g. if the loss goes `0.050 → 0.049 → 0.0489 → 0.0488...`, training has essentially converged, and more epochs won't help much.

---

## 8. What We Learned Today

| Concept                        | What It Does                                        |
| ------------------------------ | --------------------------------------------------- |
| **Neurons**              | Basic units that compute weighted sums + activation |
| **Forward Propagation**  | Data flows input → output                          |
| **Activation Functions** | Add non-linearity (ReLU is most common)             |
| **Loss Functions**       | Measure prediction error                            |
| **Gradient Descent**     | Optimizes weights to reduce error                   |
| **Training Loop**        | Repeat forward → loss → backward → update        |

---

## 9. What's Next?

In the next session, we'll learn about:

- **Backpropagation**: How gradients are computed efficiently for every weight in the network, using the chain rule

Further down the road: **Regularization** (Dropout, etc.) and **Advanced Optimizers** (Adam, RMSprop) each get their own dedicated session later in the course — see `Course_Schedule.md` for the full lineup.

---

## Key Takeaway

> Neural networks learn by adjusting millions of weights through **forward propagation**, **loss calculation**, and **gradient descent**. The **activation functions** give them the power to learn complex patterns, and the **training loop** repeats this process until the network gets good at its task.
