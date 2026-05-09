# Simulated Functional Connectivity ML Classification in Schizophrenia Using Low-Dimensional Features
Machine learning analysis of simulated functional connectivity patterns in schizophrenia using low-dimensional features. Includes connectivity simulations with Gaussian perturbations, logistic regression, cross-validation, and feature importance analysis under different noise conditions.

## Overview
This project explores whether simple low-dimensional statistical representations of simulated functional connectivity matrices can discriminate between typical and schizophrenia-like brain connectivity patterns using machine learning.
The work was inspired by recent studies investigating functional connectivity as a biomarker for schizophrenia, while also addressing an important challenge in neuroimaging research: the risk of overfitting when high-dimensional connectomes are analyzed using relatively small datasets.
Rather than relying on large-scale connectomes or deep architectures, this study investigates whether compact and interpretable statistical features can still capture meaningful discriminative information.

# Research Motivation
Recent studies have shown that functional connectivity contains relevant information for schizophrenia classification. However, high-dimensional representations may reduce interpretability and increase overfitting risk.

This project was designed as an exploratory computational study to examine:

- how simple statistical features behave under increasing noise conditions,
- whether low-dimensional representations remain informative,
- and which connectivity properties contribute most to classification performance.

The approach was partially inspired by recent work comparing functional connectivity representations in schizophrenia classification tasks.

# Methodology
## Functional Connectivity Simulation
Synthetic 20×20 symmetric connectivity matrices were generated using Gaussian perturbations.
Two groups were simulated:
- Typical control connectivity
- Schizophrenia-like connectivity

The schizophrenia-like condition included:
- increased global perturbation,
- additional stochastic noise,
- and local connectivity disorganization.
This simplified framework was intended to emulate altered connectivity variability frequently reported in schizophrenia literature.

## Feature Extraction
Two feature representations were evaluated.

### Basic representation
- Mean connectivity
- Standard deviation of connectivity

### Extended representation
- Mean connectivity
- Standard deviation
- Local vs. global connectivity difference

The goal was to compare whether adding local structural information improves classification robustness.

## Machine Learning Pipeline
The classification pipeline included:
- Logistic Regression
- StandardScaler normalization
- 5-fold cross-validation
- Permutation feature importance analysis

Noise robustness was evaluated under increasing perturbation levels:
- 0.10
- 0.15
- 0.20
- 0.25

# Results
| Noise level | Mean + Std | Mean + Std + Local |
|---|---|---|
| 0.10 | 0.997 | 0.997 |
| 0.15 | 0.948 | 0.950 |
| 0.20 | 0.838 | 0.838 |
| 0.25 | 0.760 | 0.765 |

## Main Findings
- Classification performance progressively decreased as noise increased.
- The standard deviation of connectivity emerged as the most discriminative feature.
- Local-global features contributed minimally to performance.
- Results suggest that connectivity variability may contain relevant discriminative information even in simplified representations.

### Feature robustness across noise levels - Fig. 1.png
The observed trend indicates gradual degradation of performance with noise, while preserving classification above chance levels across all conditions. This supports the robustness of low-dimensional representations.

### Model comparison at intermediate noise (σ = 0.15) - Fig. 2.png
Both feature sets exhibit comparable median performance and variability across folds. The absence of strong separation suggests limited benefit from more complex feature representations in this setting.

### Noise effect on classification performance - Fig. 3.png
Classification performance decreases gradually with increasing noise, but remains relatively stable across both feature sets. This suggests that low-dimensional connectivity descriptors preserve discriminative structure even under perturbation.

## Interpretation 
These findings align with a broader line of work in computational neuroimaging suggesting that meaningful structure in brain connectivity may be captured using low-dimensional summaries, particularly in scenarios where data is noisy or limited.
Rather than relying on high-dimensional connectomes, simpler representations may offer a more robust and interpretable alternative for machine learning applications in neuroscience.

# Feature Importance
| Feature | Importance |
|---|---|
| Mean connectivity | 0.0156 |
| Connectivity standard deviation | 0.4406 |
| Local vs global connectivity | 0.0006 |

# Technologies
- Python
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

# Limitations

This study uses simulated data rather than real neuroimaging datasets. Therefore, conclusions should be interpreted as exploratory rather than clinically generalizable.
Future work could include:
- real resting-state fMRI datasets,
- graph theoretical metrics,
- cortical gradient representations,
- EEG-derived features,
- and more advanced machine learning architectures.

# References
1. Shevchenko V. et al. *Scientific Reports* (2025), 15, 2849.
2. Alves C. L. et al. *Journal of Neural Engineering* (2023), 20, 056025.


