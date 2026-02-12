# Distributed Formation Control Driven by Recursive Equilibrium Networks Under Communication Attacks

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-blue)](https://www.sciencedirect.com/journal/knowledge-based-systems)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This repository contains the implementation of the **Adaptive Switching Control Network (ASCN)** framework for multi-UAV formation control under communication attacks, as described in our paper submitted to *Knowledge-Based Systems*.

## 📋 Overview

The ASCN framework addresses the challenge of maintaining stable multi-UAV formations in adversarial environments with degraded communication quality. Our approach introduces:

- **Communication Quality-Aware Mechanism**: Integrates physical layer (RSSI, SNR, PLR, BER) and network layer (algebraic connectivity) indicators for comprehensive assessment
- **Dual-Mode Adaptive Control**: Dynamically switches between normal mode (neighbor cooperation) and emergency mode (local state reliance) based on real-time communication quality
- **Recursive Balanced Network Architecture**: Reduces computational complexity from O(q³) to O(lq²) while maintaining control precision
- **Theoretical Stability Guarantees**: Provides exponential stability under mode switching via contraction mapping theory and average dwell time analysis

## 🎯 Key Features

### Performance Highlights
- **40-65.9% reduction** in formation tracking error compared to baseline methods
- **12× improvement** in computational efficiency versus MPC
- **Tolerates up to 70% link blockage** under DoS attacks
- **2-3× enhancement** in robustness boundaries

### Tested Scenarios
- ✅ Triangle, square, and pentagon formations (3-5 agents)
- ✅ Single-hole, double-hole, vertical, and 3D complex obstacles
- ✅ Four types of DoS attacks: Bandwidth Flooding, Selective Jamming, Protocol Disruption, Coordinated Attack
- ✅ Attack intensities from 30% to 90% blocking/degradation

## 📁 Repository Structure

```
├── main.py                 # Main simulation entry point
├── util.py                 # Utility functions (quality assessment, switching logic)
├── model.py                # ASCN controller implementation
├── README.md               # This file
└── requirements.txt        # Python dependencies (to be added upon acceptance)
```

## 🚀 Current Release

This is a **preliminary code release** containing the core implementation files:

- `main.py`: Simulation framework for multi-UAV formation control
- `util.py`: Communication quality assessment and mode switching mechanisms
- `model.py`: ASCN neural network architecture and controller design

### ⚠️ Note on Full Code Release

The **complete codebase** including:
- Training scripts and hyperparameter configurations
- Attack scenario generators
- Full evaluation benchmarks
- Visualization tools
- Pre-trained models

will be made publicly available **upon paper acceptance**.

## 🔧 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install basic dependencies (full requirements.txt coming upon acceptance)
pip install numpy scipy matplotlib torch
```

### Basic Usage

```python
# Example: Running a pentagon formation with DoS attack
from main import simulate_formation
from util import CommunicationQualityAssessor
from model import ASCNController

# Initialize 5-agent pentagon formation
config = {
    'formation_type': 'pentagon',
    'n_agents': 5,
    'attack_type': 'bandwidth_flooding',
    'attack_intensity': 0.5
}

# Run simulation
results = simulate_formation(config)
```

## 📊 Key Components

### 1. Communication Quality Assessment (`util.py`)

Comprehensive quality indicator integrating:

```python
Q(t) = w₁·Qch(t) + w₂·Qtop(t)
```

- **Channel Quality (Qch)**: Combines SNR, packet loss rate, and bit error rate
- **Topology Quality (Qtop)**: Based on algebraic connectivity λ₂(L)

### 2. ASCN Controller (`model.py`)

Dual-mode architecture with adaptive switching:

- **Normal Mode (σ=1)**: q neurons, O(lq²) complexity
- **Emergency Mode (σ=2)**: q/2 neurons, O(l(q/2)²) complexity (~75% reduction)

### 3. Mode Switching Logic

Hysteresis-based switching criteria:
- Emergency threshold: T_low = 0.4
- Recovery threshold: T_high = 0.7
- Prevents frequent oscillations near boundaries

## 📈 Experimental Results

### Formation Tracking Performance

| Method | Max Error (m) | Avg Error (m) | Steady-State Error (m) | Convergence Time (s) |
|--------|---------------|---------------|------------------------|---------------------|
| Normal-only | 8.35 | 3.24 | 0.82 | Unstable |
| Emergency-only | 4.62 | 2.15 | 0.35 | 18.5 |
| PID | 6.78 | 2.89 | 0.45 | 22.3 |
| MPC | 5.21 | 2.34 | 0.28 | 15.7 |
| **ASCN (Ours)** | **2.85** | **1.37** | **0.09** | **12.4** |

### Robustness Under Attacks

| Attack Intensity | ASCN | Normal-only | MPC | PID |
|-----------------|------|-------------|-----|-----|
| 30% DoS | 1.6m / Stable | 2.5m / Stable | 2.7m / Stable | 3.2m / Stable |
| 50% DoS | 2.9m / Stable | 8.4m / Unstable | 5.2m / Stable | 6.8m / Stable |
| 70% DoS | 4.7m / Stable | >15m / Unstable | 8.5m / Stable | 12.3m / Boundary |
| 90% DoS | 9.2m / Boundary | Divergent | 15.6m / Boundary | Divergent |

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{luan2026distributed,
  title={Distributed Formation Control Driven by Recursive Equilibrium Networks Under Communication Attacks},
  author={Luan, Tianyun and Huang, Fuzhong and Li, Jianxin and Yang, Yang},
  journal={Knowledge-Based Systems},
  year={2026},
  note={Under Review}
}
```

## 👥 Authors

- **Tianyun Luan** - School of Electronic Information Engineering, Changchun University of Science and Technology
- **Fuzhong Huang** - School of Electronic Information Engineering, Changchun University of Science and Technology
- **Jianxin Li** - School of Electronic Information Engineering, Changchun University of Science and Technology
- **Yang Yang** (Corresponding Author) - yangyang@cust.edu.cn

## 📄 License

This project will be released under the MIT License upon paper acceptance.

## 🔄 Updates

- **February 2026**: Initial code release (main.py, util.py, model.py)
- **Pending**: Full codebase release upon paper acceptance

## 📧 Contact

For questions or collaboration inquiries:
- Email: yangyang@cust.edu.cn
- Affiliation: Changchun University of Science and Technology

## 🙏 Acknowledgments

We would like to thank the reviewers for their valuable feedback and suggestions.

---

**Note**: This is a preliminary release. The complete implementation with full documentation, trained models, and reproducibility scripts will be made available following paper acceptance in *Knowledge-Based Systems*.
