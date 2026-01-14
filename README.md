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

| Node | Category | Purpose |
|------|----------|---------|
| `ECHO_ContextManager` | Framework/ECHO | 4-tier context memory with provenance tracking |
| `ECHO_ContextMerger` | Framework/ECHO | Merge contexts with priority weighting |
| `MoE_ExpertRouter` | Framework/CSQMF | Deterministic hash-based expert routing |
| `MoE_ExpertExecutor` | Framework/CSQMF | Execute with expert-specific parameters |
| `PRISM_Analyzer` | Framework/PRISM | 6-perspective reasoning analysis |
| `DeterministicSampler` | Framework/Determinism | Batch-invariant sampling configuration |
| `ChecksumValidator` | Framework/Determinism | Cryptographic reproducibility proof |
| `VFX_ShotAnalyzer` | Framework/VFX | VFX shot domain detection |

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

### ECHO Context Manager
Implements 4-tier hierarchical context memory with automatic compression:
- **Hot**: 100% precision, active context
- **Warm**: 75% precision, recent context
- **Cold**: 50% precision (NVFP4-style)
- **Archive**: 25% precision, long-term storage

### MoE Expert Router
Deterministic expert selection using hash-based routing (not random MCMC):
- **Accuracy Expert**: Low temperature (0.1), precise outputs
- **Ethics Expert**: Moderate temperature (0.3), balanced
- **Creativity Expert**: High temperature (0.8), diverse outputs
- **Compression Expert**: Low temperature (0.2), concise outputs

### PRISM Analyzer
6-perspective reasoning framework:
- Causal: What causes what?
- Optimization: How to improve?
- Hierarchical: What's the structure?
- Temporal: How does it change over time?
- Risk: What could go wrong?
- Opportunity: What possibilities exist?

### VFX Shot Analyzer
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

MIT License - See LICENSE file

## Author

Joseph Ibrahim - VFX Lighting TD
- Portfolio: [josephibrahim.com](https://josephibrahim.com)
