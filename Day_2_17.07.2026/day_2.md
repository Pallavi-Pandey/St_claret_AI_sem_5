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

### Types of Gradient Descent

| Type                    | How it Works               | Pros         | Cons                    |
| ----------------------- | -------------------------- | ------------ | ----------------------- |
| **Batch GD**      | Uses ALL data each time    | Stable       | Slow                    |
| **Stochastic GD** | Uses 1 sample at a time    | Fast         | Noisy                   |
| **Mini-Batch GD** | Uses small groups (32-256) | Best of both | Need to pick batch size |

---

## 7. The Training Loop (Putting It All Together)

The network learns by repeating these steps:

```
For each epoch (one complete pass through all data):
    For each batch of data:
        1. FORWARD PASS: Feed data through network → get prediction
        2. COMPUTE LOSS: Compare prediction to correct answer
        3. BACKWARD PASS: Calculate how to adjust weights
        4. UPDATE: Adjust weights using gradient descent
```

### Key Terms

- **Epoch**: One complete pass through the entire training data
- **Batch**: A small group of samples processed together
- **Learning Rate**: Step size for weight updates
- **Convergence**: When loss stops improving significantly

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

- **Backpropagation**: How gradients are computed efficiently
- **Advanced Optimizers**: Adam, RMSprop (better than basic gradient descent)
- **Regularization**: Techniques to prevent overfitting

---

## Key Takeaway

> Neural networks learn by adjusting millions of weights through **forward propagation**, **loss calculation**, and **gradient descent**. The **activation functions** give them the power to learn complex patterns, and the **training loop** repeats this process until the network gets good at its task.
