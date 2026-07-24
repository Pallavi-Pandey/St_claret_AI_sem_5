# Topic 4 (Part 1): Train, Test & Validation Sets, Vanishing & Exploding Gradient

## 1. Quick Recap: Backpropagation

Last session we learned **backpropagation** — the chain-rule-powered algorithm that computes the gradient of the loss with respect to every weight, so gradient descent knows exactly how to update them. We even verified our backprop code with **gradient checking**.

So far, every example trained a network and only ever looked at its own training data to judge success. Today we ask two very different questions:

1. How do we know if a network will perform well on data it has **never seen**?
2. What happens to backpropagation's chain of gradients when the network gets **deep**?

---

## 2. Why One Dataset Isn't Enough

Imagine a student who only ever practices with the exact questions that will appear on the exam, memorized word-for-word. They'd score 100% on that exact exam — but hand them a slightly different question, testing the same concept, and they'd fail. They didn't *learn*, they *memorized*.

Neural networks can do the exact same thing. A network with enough parameters can simply **memorize** its training examples instead of learning the underlying pattern. If we only ever measure performance on the data it trained on, we'd never notice.

**The fix:** keep some data completely separate from training, so we always have unseen examples to honestly check whether the network actually learned, or just memorized.

---

## 3. The Three-Way Split: Train, Validation, Test

| Set | Used For | Analogy |
| :--- | :--- | :--- |
| **Training set** | Actually updating the weights (forward pass, backward pass, update) | The practice questions a student studies from |
| **Validation set** | Checking progress *during* training, tuning hyperparameters (learning rate, architecture, etc.) | A practice test the student checks their preparation against |
| **Test set** | One final, honest check *after* all training and tuning is done — touched exactly once | The real exam, never seen until exam day |

**Why not just Train and Test?** If we used the test set to make decisions *during* training (e.g. "let's stop here because test accuracy looks good"), we'd slowly leak information from the test set into our choices — and it would stop being an honest, unseen check. The validation set exists so we can tune freely without contaminating the test set.

### Simple Example

Say we have **1,000 labeled examples** in total. A common split is **70% / 15% / 15%**:

```
Training set:    700 examples  (70%)
Validation set:  150 examples  (15%)
Test set:        150 examples  (15%)
```

We train only on the 700, repeatedly check progress on the 150 validation examples while tuning things like learning rate or number of layers, and only at the very end do we report accuracy on the 150 test examples — our one honest, final grade.

---

## 4. Underfitting vs. Overfitting

Once we have a validation set, we can compare **training loss** against **validation loss** and diagnose what's going wrong:

| Situation | Training Loss | Validation Loss | What's Happening |
| :--- | :--- | :--- | :--- |
| **Underfitting** | High | High | Model is too simple to capture the pattern at all |
| **Good Fit** | Low | Low (close to training loss) | Model learned the real underlying pattern |
| **Overfitting** | Very Low | High (and rising) | Model memorized training examples, including their noise |

**The classic overfitting signature:** as a model gets more complex (more parameters, more layers, trained longer), training loss keeps dropping — but validation loss stops improving and starts *rising* again. That rising validation loss, while training loss keeps falling, is the single clearest signal of overfitting.

*(In the hands-on notebook, we fit polynomials of increasing degree to a small noisy dataset and watch this exact pattern happen: validation error hits a minimum around degree 3, then explodes to millions by degree 14, even as training error keeps shrinking toward zero.)*

---

## 5. The Vanishing Gradient Problem

Recall from last session: backpropagation computes gradients by **multiplying** derivatives together, layer after layer, via the chain rule:

```
dLoss/dz1 = dLoss/dz2 × activation'(z2) × W2 × activation'(z1)
```

Sigmoid and Tanh activation derivatives are always **less than or equal to 1** (Sigmoid tops out at 0.25, Tanh at 1.0, and both shrink fast away from their center). When backpropagation multiplies many of these small numbers together across many layers, the result shrinks toward **zero**.

### Simple Example

Say a network has 5 layers, each using Sigmoid, and each layer's derivative happens to be at its theoretical best case, `0.25`:

```
combined gradient factor = 0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.25⁵ ≈ 0.00098
```

The gradient reaching the earliest layer is already **less than 0.1%** of its original size — and real networks are often far deeper than 5 layers, and derivatives are usually much smaller than the theoretical best case of 0.25. 

**The consequence:** the earliest layers receive a near-zero gradient. Their weights barely move during the update step (`new_weight = old_weight - learning_rate × gradient`, and if gradient ≈ 0, nothing changes). Those layers effectively **stop learning**, even while later layers keep training normally.

---

## 6. The Exploding Gradient Problem

The opposite failure mode: if weights are large (or the chosen activation doesn't squash its input, like a linear layer), the chain rule can instead **multiply numbers greater than 1** together, layer after layer — and the gradient grows exponentially instead of shrinking.

### Simple Example

Say the effective multiplier at each of 10 layers is `2` instead of a fraction:

```
combined gradient factor = 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 = 2¹⁰ = 1,024
```

The gradient reaching the earliest layer is now **1,024 times larger** than it should be.

**The consequence:** weight updates become enormous and erratic. Weights can shoot off to huge values (or `NaN`/infinity) in a single step, and the loss — instead of smoothly decreasing — spikes wildly or turns into `NaN`. Training effectively breaks.

---

## 7. Why Depth Makes Both Problems Worse

Both problems come from the exact same source: the chain rule **multiplies** one factor per layer. 

- Multiply many numbers **less than 1** → the product shrinks toward 0 (**vanishing**)
- Multiply many numbers **greater than 1** → the product grows toward infinity (**exploding**)

The more layers a network has, the more factors get multiplied together, and the more extreme the final result becomes. A shallow 2-layer network barely notices this effect; a 50-layer network can vanish or explode dramatically. This is exactly why "just add more layers" isn't automatically better — depth has to be managed carefully.

*(In the hands-on notebook, we trace a gradient backward through 20 layers under three conditions — Sigmoid with small weights, ReLU with well-scaled weights, and large weights — and the earliest-layer gradient ranges from about `10⁻⁷` to over `10¹⁸` depending purely on that setup.)*

---

## 8. Detecting These Problems in Practice

| Symptom | Likely Problem |
| :--- | :--- |
| Loss barely changes over many epochs, especially with several hidden layers | Vanishing gradient |
| Early layers' weights stay almost identical to their initial random values | Vanishing gradient |
| Loss suddenly spikes, oscillates wildly, or becomes `NaN` / `inf` | Exploding gradient |
| Training was stable, then suddenly "blows up" after one bad update | Exploding gradient |

---

## 9. A Preview of Fixes

We won't implement all of these yet, but it helps to know they exist and *why* each one targets the multiplication problem from Section 7:

- **ReLU activation** (already covered!): its derivative is exactly 1 for positive inputs, rather than a fraction — this alone fixes much of the vanishing problem in the activation function itself.
- **Better weight initialization** (e.g. scaling by `1 / √(layer size)`): keeps each layer's multiplication factor close to 1 on average, instead of consistently shrinking or growing. *(We prove this matters with real numbers in the hands-on notebook: the exact same 5-hidden-layer ReLU network either gets permanently stuck — many neurons "dying" from an oversized first update — or trains smoothly to near-zero loss, purely depending on how its weights are scaled at the start.)*
- **Gradient clipping**: if a gradient's size exceeds some threshold, simply shrink it back down before applying the update — a direct safety net against exploding gradients.
- **Regularization & Dropout**: these mainly target *overfitting* rather than vanishing/exploding gradients, and get their own deep dive next session.

---

## 10. What We Learned Today

| Concept | What It Does |
| :--- | :--- |
| **Train / Validation / Test Split** | Separates "learning," "tuning," and "honest final grading" so we never fool ourselves |
| **Underfitting** | Model too simple — high error on both training and validation data |
| **Overfitting** | Model memorized training data — training error keeps dropping while validation error rises |
| **Vanishing Gradient** | Repeated multiplication of small (<1) factors shrinks gradients toward 0 in early layers |
| **Exploding Gradient** | Repeated multiplication of large (>1) factors grows gradients toward infinity |
| **Depth** | Amplifies both problems, since more layers means more multiplications in the chain rule |

---

## 11. What's Next?

In the next session, we'll learn about:

- **Dropout**: randomly disabling neurons during training to fight overfitting
- **Regularization**: penalizing overly large weights to keep models simpler
- **Bias Correction, Learning Rate Tuning, and Softmax**: refinements that make training more reliable

---

## Key Takeaway

> A network's real performance is measured on data it has **never seen** — that's what the train/validation/test split protects. Meanwhile, backpropagation's chain rule means gradients get **multiplied** once per layer: too many small factors and they **vanish**, too many large factors and they **explode**. Both overfitting and vanishing/exploding gradients are, at their core, problems of *too much unchecked multiplication* — one in the model's capacity, the other in its gradient flow.
