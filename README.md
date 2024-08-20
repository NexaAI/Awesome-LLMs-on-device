# Awesome LLMs on Device: A Comprehensive Survey

<figure align="center">
  <img src="resources/Summary_of_on-device_LLMs_evolution.jpeg" alt="Summary of on-device LLMs’ evolution" width="800">
  <figcaption style="font-size: 10px;">Summary of On-device LLMs’ Evolution</figcaption>
</figure>

This repository contains resources and information related to our comprehensive survey paper on Large Language Models (LLMs) deployed on edge devices.

# Abstract

The advent of large language models (LLMs) has revolutionized natural language processing applications, and running LLMs on edge devices has become increasingly attractive for reasons including reduced latency, data localization, and personalized user experiences. This comprehensive review examines the challenges of deploying computationally expensive LLMs on resource-constrained devices and explores innovative solutions across multiple domains. We investigate the development of on-device LLMs, their efficient architectures, including parameter sharing and modular designs, as well as state-of-the-art compression techniques like quantization, pruning, and knowledge distillation. Hardware acceleration strategies and collaborative edge-cloud deployment approaches are analyzed, highlighting the intricate balance between performance and resource utilization. Case studies of on-device LLMs from major mobile manufacturers demonstrate real-world applications and potential benefits. The review also addresses critical aspects such as adaptive learning, multi-modal capabilities, and personalization. By identifying key research directions and open challenges, this paper provides a roadmap for future advancements in on-device LLMs, emphasizing the need for interdisciplinary efforts to realize the full potential of ubiquitous, intelligent computing while ensuring responsible and ethical deployment.

# Key Features

- Comprehensive review of on-device LLM technologies
- Analysis of efficient architectures and compression techniques
- Exploration of hardware acceleration strategies
- Case studies of real-world applications
- Discussion of future research directions and challenges

# Table of Contents

Here's a suggested organization of the references into sections based on the paper architecture:

## Foundations and Preliminaries

### Evolution of On-Device LLMs

- T. Zhang et al., "Tinyllama: An open-source small language model," arXiv preprint arXiv:2401.02385, 2024.
- X. Chu et al., "MobileVLM V2: Faster and Stronger Baseline for Vision Language Model," arXiv preprint arXiv:2402.03766, 2024.
- R. Murthy et al., "MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases," arXiv preprint arXiv:2406.10290, 2024.
- W. Chen et al., Octopus series papers (v1-v4)
- J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv preprint arXiv:2306.00978, 2024.
- S. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv preprint arXiv:2402.17764, 2024.

### LLM Architecture Foundations

- E. Frantar et al., "Gptq: Accurate post-training quantization for generative pre-trained transformers," arXiv preprint arXiv:2210.17323, 2022.
- T. Dettmers and L. Zettlemoyer, "The case for 4-bit precision: k-bit inference scaling laws," ICML, 2023.
- T. Dettmers et al., "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale," NeurIPS, 2022.
- J. Kaddour et al., "Challenges and applications of large language models," arXiv preprint arXiv:2307.10169, 2023.
- Y. Gu et al., "MiniLLM: Knowledge distillation of large language models," ICLR, 2023.

### On-Device LLMs Training

- S. Mehta et al., "OpenELM: An Efficient Language Model Family with Open Training and Inference Framework," arXiv preprint arXiv:2404.14619, 2024.

### Limitations of Cloud-Based LLM Inference and Advantages of On-Device Inference

- H. Zhang et al., "Ferret-v2: An Improved Baseline for Referring and Grounding with Large Language Models," arXiv preprint arXiv:2404.07973, 2024.
- M. Abdin et al., "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone," 2024.
- C. Saha et al., "Matrix compression via randomized low rank and low precision factorization," NeurIPS, 2023.
- Z. Yao et al., "Exploring post-training quantization in llms from comprehensive study to low rank compensation," AAAI, 2024.

### The Performance Indicator of On-Device LLMs

- G. Gerganov, "llama.cpp: Lightweight library for Approximate Nearest Neighbors and Maximum Inner Product Search," GitHub, 2023.
- Alibaba, "MNN: A lightweight deep neural network inference engine," GitHub, 2024.
- Y. Song et al., "Powerinfer: Fast large language model serving with a consumer-grade gpu," arXiv preprint arXiv:2312.12456, 2023.
- Z. Xue et al., "PowerInfer-2: Fast Large Language Model Inference on a Smartphone," arXiv preprint arXiv:2406.06282, 2024.

## Efficient Architectures for On-Device LLMs

### Model Compression and Parameter Sharing

- J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv preprint arXiv:2306.00978, 2024.
- C. Liu et al., "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases," arXiv preprint arXiv:2402.14905, 2024.

### Collaborative and Hierarchical Model Approaches

- M. Zhang et al., "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing," arXiv preprint arXiv:2405.14371, 2024.
- D. Xu et al., "Llmcad: Fast and scalable on-device large language model inference," arXiv preprint arXiv:2309.04255, 2023.

### Memory and Computational Efficiency

- B. Kim et al., "The Breakthrough Memory Solutions for Improved Performance on LLM Inference," IEEE Micro, 2024.
- S. Laskaridis et al., "MELTing point: Mobile Evaluation of Language Transformers," arXiv preprint arXiv:2403.12844, 2024.

### Mixture-of-Experts (MoE) Architectures

- R. Yi et al., "Edgemoe: Fast on-device inference of moe-based large language models," arXiv preprint arXiv:2308.14352, 2023.
- W. Yin et al., "LLM as a system service on mobile devices," arXiv preprint arXiv:2403.11805, 2024.
- J. Li et al., "Locmoe: A low-overhead moe for large language model training," arXiv preprint arXiv:2401.13920, 2024.

### General Efficiency and Performance Improvements

- Y. Park et al., "Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs," arXiv preprint arXiv:2402.10517, 2024.
- Z. Yan et al., "On the viability of using llms for sw/hw co-design: An example in designing cim dnn accelerators," IEEE SOCC, 2023.

## Model Compression and Optimization Techniques for On-Device LLMs

### Quantization

- E. Frantar et al., "Gptq: Accurate post-training quantization for generative pre-trained transformers," arXiv preprint arXiv:2210.17323, 2022.
- J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," arXiv preprint arXiv:2306.00978, 2024.
- T. Dettmers et al., "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale," NeurIPS, 2022.
- S. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv preprint arXiv:2402.17764, 2024.

### Pruning

- J. Kaddour et al., "Challenges and applications of large language models," arXiv preprint arXiv:2307.10169, 2023.

### Knowledge Distillation

- Y. Gu et al., "MiniLLM: Knowledge distillation of large language models," ICLR, 2023.

### Low-Rank Factorization

- R. Saha et al., "Matrix compression via randomized low rank and low precision factorization," NeurIPS, 2023.
- Z. Yao et al., "Exploring post-training quantization in llms from comprehensive study to low rank compensation," AAAI, 2024.

## Hardware Acceleration and Deployment Strategies

### Popular On-Device LLMs Framework

- Various frameworks (llama.cpp, MNN, PowerInfer, ExecuTorch, MediaPipe, MLC-LLM, VLLM, OpenLLM)

### Hardware Acceleration

- J. Kim et al., "Aquabolt-XL: Samsung HBM2-PIM with in-memory processing for ML accelerators and beyond," IEEE Hot Chips, 2021.
- B. Kim et al., "The Breakthrough Memory Solutions for Improved Performance on LLM Inference," IEEE Micro, 2024.
- J. Kim et al., "Aquabolt-XL: Samsung HBM2-PIM with in-memory processing for ML accelerators and beyond," IEEE Hot Chips, 2021.

# Model Reference

|         Model         |      Institute      | Paper                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-------------------: | :-----------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      Gemini Nano      |       Google        | [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/pdf/2312.11805.pdf)                                                                                                                                                                                                                                                                                                                          |
| Octopus series model  |       Nexa AI       | [Octopus v2: On-device language model for super agent](https://arxiv.org/pdf/2404.01744.pdf)<br>[Octopus v3: Technical Report for On-device Sub-billion Multimodal AI Agent](https://arxiv.org/pdf/2404.11459.pdf)<br>[Octopus v4: Graph of language models](https://arxiv.org/pdf/2404.19296.pdf)<br>[Octopus: On-device language model for function calling of software APIs](https://arxiv.org/pdf/2404.01549.pdf) |
| OpenELM and Ferret-v2 |        Apple        | OpenELM is a significant large language model integrated within iOS to enhance application functionalities. <br>Ferret-v2 significantly improves upon its predecessor, introducing enhanced visual processing capabilities and an advanced training regimen.                                                                                                                                                          |
|      Phi series       |      Microsoft      | [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219.pdf)                                                                                                                                                                                                                                                                                                 |
|        MiniCPM        | Tsinghua University | [A GPT-4V Level Multimodal LLM on Your Phone](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)                                                                                                                                                                                                                                                                                                                    |
|       Gemma2-9B       |       Google        | [Gemma 2: Improving Open Language Models at a Practical Size](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)                                                                                                                                                                                                                                                                                 |
|      Qwen2-0.5B       |    Alibaba Group    | [Qwen Technical Report](https://arxiv.org/pdf/2309.16609.pdf)                                                                                                                                                                                                                                                                                                                                                         |

# Tutorial:

- MIT: [TinyML and Efficient Deep Learning Computing](https://efficientml.ai)
- Harvard: [Machine Learning Systems](https://mlsysbook.ai/)

# Citation

If you find this survey helpful, please consider citing our paper:
