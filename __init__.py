"""
ComfyUI Framework Ecosystem
===========================
AI framework integration nodes for ComfyUI.

Frameworks:
- ECHO 2.0: 4-tier context memory
- CSQMF-R1: MoE expert routing
- PRISM: Multi-perspective analysis
- ThinkingMachines: Deterministic inference

Nodes:
- ECHO_ContextManager
- ECHO_ContextMerger
- MoE_ExpertRouter
- MoE_ExpertExecutor
- PRISM_Analyzer
- DeterministicSampler
- ChecksumValidator
- VFX_ShotAnalyzer
"""

__version__ = "1.0.0"
__author__ = "Joseph Ibrahim"

from .comfyui_framework_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = None
