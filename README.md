
# Facial Recognition Using Singular Value Decomposition (SVD) with HPC Integration

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=CuteLoop/faces)

This repository contains the implementation of a scalable facial recognition system using **Singular Value Decomposition (SVD)** in **MATLAB** with **High-Performance Computing (HPC)** integration. The project demonstrates the use of parallel computation for dimensionality reduction, feature extraction, and classification of facial images from publicly available datasets.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation and Setup](./usage.md)
5. [Usage](./usage.md)
6. [Roadmap](./Roadmap.md) and [Roadmap2](./Roadma2p.md)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## Introduction
Facial recognition systems are vital in applications such as security, authentication, and human-computer interaction. This project leverages **SVD** for dimensionality reduction and feature extraction, combined with MATLAB's Parallel Computing Toolbox to handle large datasets efficiently.

### Key Objectives:
- Implement **SVD** for feature extraction.
- Utilize **HPC resources** for computationally intensive tasks.
- Train and test a supervised classification model for face recognition.
- Evaluate computational efficiency, accuracy, and scalability.

---

## Features
- **Dataset Preprocessing**: Grayscale conversion, resizing, and normalization.
- **SVD-Based Dimensionality Reduction**: Retain top components for efficiency.
- **Supervised Classification**: Supports models like SVM and k-NN.
- **Parallel Processing**: Integrates MATLAB's HPC capabilities for scalability.

---

## Requirements
- **MATLAB** (R2023b or newer)
  - Parallel Computing Toolbox
- Access to an HPC environment (e.g., SLURM cluster)
- Public facial datasets:
  - [Extended Yale B Dataset](https://academictorrents.com/details/06e479f338b56fa5948c40287b66f68236a14612)
  - [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)


## Acknowledgments
This project acknowledges the contributions of the following:
- **ChatGPT** by OpenAI for assistance in planning, structuring, and documentation of this repository.
- **Marek Rychlik** for guidance and resources in HPC and MATLAB.
- Public datasets [Extended Yale B Dataset](https://academictorrents.com/details/06e479f338b56fa5948c40287b66f68236a14612) and [LFW](http://vis-www.cs.umass.edu/lfw/).

---

## License
© 2024 [Your Name or Institution]. All rights reserved.

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Disclaimer
This project was developed as part of **Math 589A Algorithms and Numerical Methods Final Project** and is intended for educational purposes only.

