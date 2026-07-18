# Intro to Neural Network & Deep Learning: Understanding Human Brain's functionality, Gradient Descent Optimization

## 1. What is a Neural Network?
The term **Neural Network** draws its inspiration from the biological architecture of the human brain. "Neural" refers to the brain's neurons, and "Network" suggests how these cellular units are extensively interconnected to process and transmit information. 

In Artificial Intelligence and Machine Learning, a Neural Network is a mathematical computing model consisting of interconnected nodes (referred to as neurons) organized into structured layers.

### The Anatomy of an Artificial Neuron
At a microscopic level within the model, a **Neuron** simply acts as a container holding a single number, known as an **Activation** value. 
- This activation value generally ranges between **0 and 1**.
- An activation of **1.0** indicates that the neuron is highly active or "firing."
- An activation of **0.0** means the neuron is completely inactive.

Overall, the entire network acts as a highly complicated mathematical function. It intakes a structured array of numbers (the input data) and processes them through layers to produce a final set of numbers (the output prediction).

## 2. Architecture of a Multi-Layer Perceptron (MLP)
A Multi-Layer Perceptron (MLP) is a foundational, "plain-vanilla" type of neural network. A classic educational example for an MLP is a network designed to recognize handwritten digits (such as reading the numbers 0-9 written on a piece of paper).

### Breakdown of Network Layers:
1. **Input Layer**: The very first layer that receives the raw unstructured data. 
   - *Example*: Imagine a grayscale image of a handwritten digit formatted as 28 pixels by 28 pixels. This gives us `28 × 28 = 784` pixels in total. Each pixel is fed into its own neuron in the network (e.g., holding an activation of 0 for a black pixel, and 1 for a white pixel). Thus, the input layer consists exclusively of these **784 neurons**.
   
2. **Hidden Layers**: The intermediate layers sitting between the input and output. The dimensions here (how many layers and how many neurons per layer) are arbitrary choices made by the network designer.
   - For instance, we might choose 2 hidden layers, each containing 16 neurons.
   - **How they work**: Activations in one layer determine the activations for the neurons in the subsequent layer. 
   - **What they do**: The hidden layers work to recognize sub-components. For example, to recognize a handwritten "9", a hidden layer might identify two geometric components: a round loop at the top and a vertical stroke on the right side. Similarly, an "8" is recognized as two stacked round circles. The deepest hidden layers break these shapes down even further into simple **edges** and lines.

3. **Output Layer**: The terminal layer that gives the network's final prediction.
   - *Example*: In our digit recognizing network, this layer has exactly **10 neurons**, each representing a digit from 0 to 9. The neuron with the brightest (highest) activation value represents the network's confident choice.

## 3. Parameters: Weights and Biases (The Brain's Logic)
Neurons don't just pass numbers to each other blindly; they do so through governed connections. During the learning phase, the network assigns specific numbers to every connection:

- **Weights**: These are numerical values assigned to every connection passing from one layer to the next. 
  - They can be positive (increasing the next neuron's activation) or negative (suppressing the activation). 
  - **Purpose**: Weights dictate exactly what geometric pattern or feature (like a horizontal edge) the subsequent neurons should be looking for. When the target pattern is found in the pixels, the positive weights heavily stimulate the corresponding hidden neuron.
  
- **Bias**: An additional threshold number added to the total weighted sum arriving at a neuron. 
  - **Purpose**: The bias essentially dictates how strictly high the weighted sum needs to be before a neuron is allowed to trigger and become active. It filters out low-intensity "noise" and prevents unwarranted activations unless a distinct threshold is passed.

### The Massive Scale of Parameters
The interconnections in a neural network are dense. Every neuron in a given layer is connected to *every* neuron in the previous layer. For our simple network parsing handwritten digits (784-input → 16-hidden → 16-hidden → 10-output), the math scales rapidly:
- The first layer of connections alone accounts for `784 neurons × 16 neurons = 12,544 weights`. Add the `16` biases for the second layer, giving **12,560** parameters just for step one.
- The entire lightweight MLP network totals **13,002 parameters**. During computation, the network evaluates this using rapid **Matrix Multiplication**.

## 4. Activation Functions
When calculating the raw weighted sum combined with biases, the resulting number can easily exceed 1 or fall far into the negatives. Since neuron activations need to stay between 0 and 1, we use an **Activation Function** to compress these outputs into the intended bounds.

- **Sigmoid Function**: A mathematical S-curve function that squashes any real number strictly into a range between 0 and 1. Very negative raw values are compressed near 0, and large positive values are compressed near 1.
- **ReLU (Rectified Linear Unit)**: Historically important, the Sigmoid function eventually proved to be a computational bottleneck because it acts as a "slow learner" during training. The modern industry standard has overwhelmingly shifted towards **ReLU**, an activation function that is far more computationally efficient while achieving better learning convergence.

## 5. What Does "Learning" Actually Mean?
In the context of Machine Learning, **Learning** means automatically determining the correct set of values for the thousands of weights and biases in the network. 

Attempting to tune 13,002 interconnected weights manually is functionally impossible for a human being. However, by supplying the machine with thousands of example images alongside their correct answers (labelled data), the network iteratively alters its own parameters until it can consistently perform the desired task (such as flawlessly predicting handwritten digits).

## 6. Real-World Power: Molecule Generation
Beyond simple image recognition, the concepts underpinning Neural Networks scale to extraordinary heights. Modern AI—utilizing **Deep Generative Models**, **Large Language Models (LLMs)**, and **Diffusion Models**—is actively accelerating fields like computational chemistry. 

These models are deployed to explore chemical spaces with an estimated 10⁶⁰ potential compounds. By efficiently generating and testing synthetic structures, AI can discover novel compounds exhibiting target physical, biological, or pharmacological properties, revolutionizing the pace of modern drug discovery and advanced materials science.