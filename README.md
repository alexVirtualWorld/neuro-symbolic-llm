[English](README.md) | [中文](README_zh-CN.md)

# Extreme Compute Reduction via Neuro-Symbolic Hybrid Architecture and Logic Distillation: Engineering Implementation on Consumer Hardware

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23090915.svg)](https://doi.org/10.5281/zenodo.23090915)

> **Note:** The initial preprint paper is now available on Zenodo. We are continuously refining the document and will release subsequent versions.

## Abstract
As Large Language Models (LLMs) evolve according to scaling laws, the compute consumption and memory wall bottlenecks in complex logical reasoning tasks become increasingly prominent. Particularly on resource-constrained consumer hardware (e.g., 12GB VRAM GPUs), maintaining high-concurrency logical inference is difficult to achieve through pure neural network architectures. This paper proposes a decoupled Neuro-Symbolic Architecture that physically isolates non-deterministic semantic parsing from deterministic logical operations via an "LLM Intent Routing + Symbolic Engine Execution" paradigm. For closed-domain logic scenarios, we further propose a "Logic-to-Tree Distillation" method, which exhausts the state space through orthogonal experimental design to downgrade and compile the LLM's deductive capabilities into static decision trees. Theoretical analysis and preliminary system implementation demonstrate that, while ensuring 100% deterministic accuracy in logical reasoning, this architecture reduces runtime GPU compute overhead by over 90% and completely eliminates the Out-Of-Memory (OOM) risks associated with Key-Value (KV) Cache expansion in long logical chains.

---

## 1. Introduction

Current auto-regressive Transformer architectures have achieved breakthroughs in natural language understanding but possess inherent engineering flaws when handling intensive mathematical computations and multi-step causal logic. LLMs approximate discrete logical boundaries using high-dimensional continuous vector spaces, which not only leads to probabilistic logic hallucinations but also consumes floating-point operations (FLOPs) disproportionate to task complexity during inference.

In constrained hardware environments (e.g., edge devices or single consumer-grade GPUs), traditional optimization paths (such as Mixture of Experts or dynamic 2D block routing) can reduce single-activation compute but still face extreme memory bandwidth consumption and the compute paradox of routing networks. To break through this physical limit, this study abandons the traditional approach of seeking solutions entirely within neural networks. Instead, it introduces neuro-symbolics, downgrading the LLM to a pure Natural Language to Abstract Syntax Tree (NL-to-AST) semantic translator, completely offloading the heavy computational payload to a zero-VRAM CPU symbolic engine.

---

## 2. Methodology

The proposed system architecture is divided into two core execution phases: Online Hybrid Pipeline and Offline Logic Distillation.

### 2.1 LLM-to-Symbolic Interface (LSI) Specification

In the online hybrid mode, the system defines a strict LSI boundary protocol. The model (e.g., Qwen2.5-3B/7B Quantized) does not participate in any numerical calculations; its sole physical function is feature extraction.

- **Input Constraints and Defensive Deserialization:** By injecting JSON Schema constraints via low-level hardware drivers (e.g., `llama.cpp`), the system truncates invalid probability distributions at the Logits layer, ensuring highly structured JSON output. The symbolic layer employs defensive parsing strategies upon receiving data, dynamically reconstructing non-standard strings into memory-safe numerical tensors or scalars.
- **Computational Complexity Bifurcation:** The time complexity of semantic parsing is fixed at $O(L)$ (where $L$ is the input sequence length) and is handled by the GPU. In contrast, tasks like matrix determinant calculation with a complexity of $O(N^3)$ are taken over by traditional industrial low-level libraries like LAPACK/BLAS on the CPU.

### 2.2 Logic-to-Tree Distillation

For closed-domain business scenarios with finite dimensions, this paper proposes an offline distillation scheme that completely eliminates runtime GPU load. The mathematical model and engineering steps are as follows:

1. **State Space Discretization:** Map continuous input variables to finite discrete intervals to construct the input feature set $\mathbf{X} = \{x_1, x_2, \dots, x_n\}$.
2. **Offline High-Concurrency Annotation:** Utilize GPU compute to generate the Cartesian product matrix of the features. Batch Prompts force the LLM to output corresponding decision labels $\mathbf{Y}$. For broad state spaces facing the curse of dimensionality, Pairwise Testing algorithms are used for uniform sparse sampling to keep the annotation volume within polynomial time.
3. **Decision Tree Compilation:** Feed the generated supervised dataset $(\mathbf{X}, \mathbf{Y})$ into information entropy gain algorithms (e.g., CART or C4.5) to generate static behavior trees.
4. **Hardware-Layer Deployment:** Hardcode the compiled rule trees directly into $O(1)$ complexity Lookup Tables or multi-condition jump instructions, achieving microsecond-level inference on ultra-low power CPUs.

---

## 3. Experimental Setup

To verify the engineering value of this architecture on consumer hardware, this study designed the following quantitative evaluation pipeline. Experiments were conducted on a hardware platform equipped with an RTX 4070 Ti (12GB VRAM).

### 3.1 Evaluation Metrics

- **Intent Routing Accuracy:** Evaluates the ability of small-parameter LLMs (3B-7B scale) to accurately identify user natural language intents and extract lossless parameters under JSON constraints.
- **E2E Deterministic Accuracy:** Compares the logical accuracy of standard Chain-of-Thought (CoT) LLM prompting against the final output correctness of the neuro-symbolic architecture (theoretical value: 100%).
- **Resource Profiling:**
  - **Energy Consumption:** Monitors and compares the cumulative GPU power consumption (Joules) in a single query.
  - **Peak VRAM:** Quantifies the difference between the KV Cache expansion rate during long logical chain deduction and the fixed VRAM footprint of this architecture.

### 3.2 Offline Distillation Benchmarking

Select standard closed-domain decision tasks with multi-dimensional conditional branches. Record the offline GPU compute time required for complete state space exhaustion and the static byte size of the compiled rule table in CPU memory to demonstrate the dimensionality reduction impact in complex closed-domain scenarios.

## 4. Experimental Results and Performance Analysis

This study completed automated benchmark testing on an NVIDIA GeForce RTX 4070 Ti platform equipped with 12GB VRAM. The test set included natural language matrix determinant calculations and structured SQL data queries. The core quantitative data is shown below:

| Evaluation Metric | Result | Physical Constraint |
| --- | --- | --- |
| Intent Routing Accuracy | 100.00% | JSON Schema Logits Constraint |
| E2E Deterministic Accuracy | 100.00% | CPU Deterministic Library (LAPACK) |
| Avg LLM Latency | 943.81 ms | Qwen2.5-3B-Instruct (Q6_K) |
| Avg Symbolic Latency | 0.05 ms | Pure CPU Execution |
| Peak VRAM Usage | 4.0 GB | Full Hardware GPU Offload |

### 4.1 Accuracy and Hallucination Elimination Boundary

Aggregated data from multiple benchmarks indicates that the LSI-based neuro-symbolic architecture achieved a flawless 100.00% accuracy in both routing recognition and end-to-end execution. Through JSON Schema Logits constraints, the model completely abandons probability-based mathematical deduction, strictly confining its physical role to the feature extraction layer, thereby fundamentally eliminating LLM hallucination phenomena in complex logical reasoning.

### 4.2 Compute Latency and Compute Offloading Efficiency

In terms of latency, the system's compute bifurcation mechanism demonstrates extreme asymmetry. The average intent routing time for the LLM (Qwen2.5-3B-Instruct Q6_K) stabilized between 1006.52 ms and 1062.84 ms. Once the task flowed to the CPU symbolic engine, the complex mathematical matrix calculation time plummeted to just 0.05 ms. Simultaneously, GPU utilization showed explicit compute peaks of 54%, 55%, and 58% during model inference, objectively proving that the CUDA hardware interface successfully took over the intent parsing task, while subsequent logical deductions achieved 100% GPU compute offloading.

### 4.3 Engineering Breakthrough of the Memory Wall

Under the severe 12GB VRAM constraint, the system's "Dedicated GPU Memory" footprint remained extremely stable, strictly locked at a peak of 4.0/12.0 GB (including base OS overhead, dipping as low as 3.7 GB). Because logical calculations are completely stripped to the CPU, the Transformer model does not need to maintain massive multi-step logical chain states (KV Cache) during inference. This architecture reserved up to 8GB of VRAM safety redundancy for the graphics card, thoroughly preventing Out-Of-Memory (OOM) crashes caused by long-context deduction.

### 4.4 Offline Distillation Overhead and State Space Compression

To validate the engineering feasibility of the "Logic-to-Tree Distillation" method in closed-domain scenarios, we conducted an exhaustive state space enumeration for a mixed intent parsing task (mathematical computation and structured queries) on consumer-grade hardware (NVIDIA RTX 4070 Ti, 12GB VRAM). The experiment defined a four-dimensional feature space generating 864 orthogonal state combinations. The Qwen2.5-3B model, constrained by grammar-level JSON schemas for deterministic outputs, was utilized for offline annotation. The quantitative results are presented in Table 2.

**Table 2: Quantitative Metrics of Offline Logic Distillation**

| Metric | Measured Value |
| :--- | :--- |
| **Closed-Domain Task** | Mixed Intent Parsing (Math & SQL) |
| **State Space Size** | 864 feature combinations (6 × 6 × 6 × 4) |
| **Avg. LLM Inference Latency** | 618.3 ms |
| **Total Offline Distillation Time** | 534.24 seconds (~8.9 minutes) |
| **Static Lookup Table Size** | 136.53 KB |
| **Online Routing Latency** | < 0.05 ms ($O(1)$ complexity) |
| **Online GPU/VRAM Overhead** | 0 MB (100% CPU execution) |

The experimental data demonstrates that while the LLM requires an average of 618.3 ms for a single structured intent routing pass, the exhaustive annotation of 864 state combinations can be completed offline in approximately 8.9 minutes. This process successfully compiles the deductive logic of the LLM into a highly compact 136.53 KB static hash table. During online deployment, the CPU executes an $O(1)$ key-value lookup, permanently locking the intent parsing latency under 0.05 ms and entirely eliminating GPU compute overhead. This confirms the overwhelming advantage of the offline distillation strategy for achieving ultra-fast, high-concurrency deterministic reasoning on resource-constrained devices.

### 5. Conclusion

This paper proposes and implements a neuro-symbolic hybrid inference architecture to address the scaling law bottlenecks of LLMs in complex deterministic reasoning tasks. The architecture's engineering value is validated across two core execution phases:

For dynamic online routing, the LLM-to-Symbolic Interface (LSI) strictly confines the LLM to feature extraction via JSON constraints. On an RTX 4070 Ti, this online pipeline maintains a rigid peak VRAM footprint of 4.0 GB, completely eliminating the OOM risks of long-context KV Cache expansion while achieving 100% end-to-end logical accuracy.

For closed-domain scenarios, we further introduced a "Logic-to-Tree Distillation" mechanism. Empirical results confirm that an exhaustive state space of 864 intent combinations can be compiled offline in under 9 minutes, generating an ultra-lightweight 136.5 KB static rule set. During online execution, this compiled symbolic engine operates at $O(1)$ complexity, permanently locking routing latency below 0.05 ms and entirely eliminating runtime GPU compute overhead. 

Ultimately, this paradigm proves that by decoupling semantic parsing from deductive calculation—and further downgrading LLMs into offline logic compilers when applicable—consumer-grade GPUs possess the physical potential to host zero-hallucination, high-concurrency deterministic AI applications.
