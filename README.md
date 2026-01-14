# ComfyUI Framework Ecosystem

AI framework integration nodes for ComfyUI. Implements cutting-edge AI research aligned with NVIDIA CES 2026 announcements.

## Installation

### Option 1: ComfyUI Manager
Search for "Framework Ecosystem" in ComfyUI Manager and install.

### Option 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/josephibrahim/comfyui-framework-ecosystem.git
pip install numpy
```

## 8 Custom Nodes

| Display Name | Category | What It Does |
|--------------|----------|--------------|
| **Memory Bank** (Hot→Cold) | JI/Memory | Store context across 4 tiers with auto-compression |
| **Merge Memories** | JI/Memory | Combine contexts with priority weighting |
| **Smart Model Picker** | JI/Routing | Pick the right AI expert for your task |
| **Run Expert** | JI/Routing | Execute with expert-specific parameters |
| **Multi-Angle Analyzer** | JI/Analysis | Analyze from 6 different perspectives |
| **Locked Sampler** | JI/Reproducible | Same seed = same output, guaranteed |
| **Output Matcher** | JI/Reproducible | Verify your outputs match exactly |
| **Shot Type Detector** | JI/VFX | Detect pyro/fluid/RBD/cloth/hair/lighting |

## CES 2026 Alignment

These nodes implement frameworks that directly map to NVIDIA's CES 2026 announcements:

| Framework | CES 2026 Announcement | Alignment |
|-----------|----------------------|-----------|
| **ECHO 2.0** | Context Memory Platform | 4-tier memory, NVFP4 compression |
| **CSQMF-R1** | Multi-Model Agents | Deterministic expert routing |
| **PRISM** | Alpamayo Reasoning | Multi-perspective analysis |
| **ThinkingMachines** | Reproducible Inference | Batch-invariant determinism |

## The Critical Determinism Fix

```python
# The ThinkingMachines discovery:
# temperature=0 is NOT enough for reproducibility
# Batch size variance causes different outputs!

batch_size = 1  # NEVER CHANGE THIS
cudnn_deterministic = True
cudnn_benchmark = False  # Disable auto-tuning variance
```

## Node Details

### Memory Bank (ECHO Context Manager)
Implements 4-tier hierarchical context memory with automatic compression:
- **Hot**: 100% precision, active context
- **Warm**: 75% precision, recent context
- **Cold**: 50% precision (NVFP4-style)
- **Archive**: 25% precision, long-term storage

### Smart Model Picker (MoE Expert Router)
Deterministic expert selection using hash-based routing (not random MCMC):
- **Accuracy Expert**: Low temperature (0.1), precise outputs
- **Ethics Expert**: Moderate temperature (0.3), balanced
- **Creativity Expert**: High temperature (0.8), diverse outputs
- **Compression Expert**: Low temperature (0.2), concise outputs

### Multi-Angle Analyzer (PRISM)
6-perspective reasoning framework:
- Causal: What causes what?
- Optimization: How to improve?
- Hierarchical: What's the structure?
- Temporal: How does it change over time?
- Risk: What could go wrong?
- Opportunity: What possibilities exist?

### Shot Type Detector (VFX Shot Analyzer)
VFX-specific domain detection for routing to specialists:
- Pyro (fire, smoke, explosions)
- FLIP (water, fluids)
- RBD (destruction, rigid bodies)
- Cloth (fabric, softbody)
- Hair (fur, grooming)
- Lighting (renders, shaders)
- Comp (compositing)

## Requirements

- ComfyUI
- numpy >= 1.21.0
- torch >= 2.0.0 (optional, for CUDA determinism)

## License

**Dual-licensed under AGPL-3.0 and Commercial licenses.**

| Use Case | License | Requirements |
|----------|---------|--------------|
| Open source projects | AGPL-3.0 | Release derivatives under AGPL-3.0 |
| Personal/educational | AGPL-3.0 | Attribution required |
| SaaS/proprietary | Commercial | [Contact for license](mailto:joseph@josephibrahim.com) |

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for commercial terms.

**Why dual-license?** These nodes contain novel AI framework IP (ECHO, CSQMF, PRISM). AGPL ensures the open-source community benefits from improvements while commercial licensing enables proprietary use.

## Author

Joseph Ibrahim - VFX Lighting TD
- Portfolio: [josephibrahim.com](https://josephibrahim.com)
- Commercial inquiries: joseph@josephibrahim.com
