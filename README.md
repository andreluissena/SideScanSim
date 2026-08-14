# SideScanSim

**A physically-grounded, open-source synthetic side-scan sonar image simulator with automatic ground truth generation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red.svg)](https://streamlit.io/)

---

## Overview

SideScanSim generates realistic synthetic side-scan sonar (SSS) images from first-principles acoustic physics, providing automatic pixel-level ground truth labels for supervised machine learning.

Unlike game-engine or deep-learning-based approaches, SideScanSim encodes the physical formation process explicitly:

- **Lambert backscatter model** for seafloor intensity (with BS₀ per sediment class)
- **Vectorized ray casting** for acoustic shadow computation
- **Transmission loss** via spherical spreading + Francois-Garrison absorption
- **Coherent speckle** (Rayleigh/exponential model)
- **TVG compensation** (Time-Varying Gain)

The simulator covers four operational domains: environmental monitoring, port/infrastructure inspection, naval defense (mine detection), and search & rescue.

---

## Key features

- Parametric control over sonar, seafloor, environment, and object parameters
- Automatic ground truth: shadow mask, bounding box, Target Strength, grazing angle, and shadow length per object
- 10 sediment classes (rock → soft mud) with physically-calibrated BS₀ values
- 9 object materials (steel → rubber) with physically-grounded Target Strength values
- 6 object geometries (sphere, box, cylinder, mound, rock, wreck)
- Interactive Streamlit GUI with real-time lateral geometry diagram and intensity profile
- Display-layer brightness/contrast control (percentile stretch + gamma) fully decoupled from physical data
- Reproducible output via explicit random seeds (speckle + texture)
- PNG export with scene parameters encoded in filename

---

## Installation

```bash
git clone https://github.com/andreluissena/SideScanSim.git
cd SideScanSim
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Quick start

1. Set **Sonar** parameters (frequency, range, altitude)
2. Choose a **Seafloor** sediment type
3. Optionally add an **Object** (shape, material, position, height)
4. Click **🔄 Gerar imagem**
5. Inspect the synthetic image, geometry diagram, and ground truth metrics
6. Download with **💾 Salvar imagem (PNG)**

See [`docs/SideScanSim_Tutorial.pdf`](docs/SideScanSim_Tutorial.pdf) for the complete step-by-step user guide.

---

## Physical model

The sonar equation implemented is:

```
RL = SL − 2·TL + BS + 10·log₁₀(A)
```

Where:

| Term | Formula | Description |
|---|---|---|
| TL | `20·log₁₀(r) + α·r/1000` | Spherical spreading + Francois-Garrison absorption |
| BS(θ) | `BS₀ + 20·log₁₀(sin θ)` | Lambert's Law backscatter |
| L (shadow) | `H·x / (h − H)` | Ray casting geometry |

Full model documentation: [`docs/01_physics_reference.md`](docs/01_physics_reference.md)

---

## Repository structure

```
SideScanSim/
├── app.py                        # Streamlit parametric GUI
├── sidescan_engine_v4.py         # Acoustic simulation engine
├── requirements.txt              # Python dependencies
├── docs/
│   ├── SideScanSim_Tutorial_EN.pdf  # Step-by-step user guide
└── examples/
    └── (sample synthetic images)
```

---

## Sediment classes and object materials

### Sediment classes (BS₀ reference values)

| Sediment | BS₀ [dB] | Visual appearance |
|---|---|---|
| Rock | −8 | Very bright, rough texture |
| Coarse gravel | −15 | Bright, high local variance |
| Coarse sand | −22 | Moderately bright |
| Medium sand | −25 | Medium gray, smooth texture |
| Fine sand | −28 | Darker gray, low variance |
| Sandy mud | −33 | Dark, relatively uniform |
| Soft mud | −38 | Very dark, almost no texture |

### Object materials (Target Strength)

| Material | TS [dB] | Highlight appearance |
|---|---|---|
| Steel | +22 | Very bright (near white) |
| Aluminum | +20 | Very bright |
| Concrete | +10 | Moderately bright |
| Natural rock | +8 | Moderate |
| Wood | +2 | Low |
| Plastic | −2 | Very low |
| Rubber | −8 | Almost no highlight |

---

## Citation

If you use SideScanSim in your research, please cite:

```bibtex
@inproceedings{sena2026sidescansim,
  title     = {SideScanSim: A Physically-Grounded Synthetic Side-Scan Sonar
               Image Simulator for Machine Learning Dataset Generation},
  author    = {Sena, André Luis S. and others},
  booktitle = {Anais do ERBASE 2026 / WEIBASE},
  year      = {2026},
  note      = {Software available at https://github.com/andreluissena/SideScanSim}
}
```

*A journal article with expanded validation and comparative analysis is in preparation (IEEE Journal of Oceanic Engineering).*

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

**André Luis S. Sena**
LABTAO — Laboratório de Tecnologia Atmosférica e Oceânica
Federal University of Bahia (UFBA), Salvador, Brazil
andresena@ufba.br
https://github.com/andreluissena
