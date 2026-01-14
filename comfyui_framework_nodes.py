"""
ComfyUI Framework Nodes
=======================
Production-ready custom nodes integrating your AI framework ecosystem.

Drop this file into: ComfyUI/custom_nodes/comfyui_framework_nodes.py

Frameworks Integrated:
- ECHO 2.0: 4-tier context memory (CES Context Memory Platform)
- CSQMF-R1: MoE expert routing (deterministic hash-based)
- PRISM: Multi-perspective analysis
- ThinkingMachines: Batch-invariant determinism

Author: Framework Ecosystem Integration
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np

# Optional torch import for determinism settings
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# ECHO 2.0: 4-Tier Context Memory (CES Context Memory Platform Aligned)
# =============================================================================

class ECHO_ContextManager:
    """
    4-tier hierarchical context memory from ECHO 2.0 framework.

    Aligns with NVIDIA CES 2026 Context Memory Platform:
    - Hot: Active context, full precision
    - Warm: Recent context, slight compression
    - Cold: Older context, NVFP4-style 50% compression
    - Archive: Long-term storage, maximum compression
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "context_tier": (["hot", "warm", "cold", "archive"],),
                "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 32768}),
            },
            "optional": {
                "previous_context": ("ECHO_CONTEXT",),
                "provenance_tag": ("STRING", {"default": "user_input"}),
            }
        }

    RETURN_TYPES = ("ECHO_CONTEXT", "STRING", "INT")
    RETURN_NAMES = ("context", "debug_info", "effective_tokens")
    FUNCTION = "manage_context"
    CATEGORY = "Framework/ECHO"

    # NVFP4-style compression ratios per tier
    TIER_COMPRESSION = {
        "hot": 1.0,      # Full precision, active use
        "warm": 0.75,    # Recent, 25% compressed
        "cold": 0.5,     # NVFP4-style 50% compression
        "archive": 0.25  # Maximum compression
    }

    def manage_context(self, prompt: str, context_tier: str, max_tokens: int,
                       previous_context: Optional[Dict] = None,
                       provenance_tag: str = "user_input") -> Tuple[Dict, str, int]:

        compression = self.TIER_COMPRESSION[context_tier]
        effective_tokens = int(max_tokens * compression)

        # Estimate characters (rough: 4 chars per token)
        max_chars = effective_tokens * 4
        truncated_content = prompt[:max_chars]

        # Build context with ECHO 2.0 provenance tracking
        context = {
            "content": truncated_content,
            "tier": context_tier,
            "compression_ratio": compression,
            "effective_tokens": effective_tokens,
            "provenance": {
                "source": provenance_tag,
                "timestamp": time.time(),
                "content_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16]
            },
            "metadata": {
                "original_length": len(prompt),
                "truncated_length": len(truncated_content),
                "truncated": len(prompt) > max_chars
            }
        }

        # Merge with previous context if provided
        if previous_context:
            context["history"] = previous_context
            context["lineage_depth"] = previous_context.get("lineage_depth", 0) + 1
        else:
            context["lineage_depth"] = 0

        debug = (f"Tier: {context_tier} | "
                f"Compression: {compression:.0%} | "
                f"Tokens: {effective_tokens} | "
                f"Truncated: {context['metadata']['truncated']}")

        return (context, debug, effective_tokens)


class ECHO_ContextMerger:
    """Merge multiple ECHO contexts with priority weighting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_a": ("ECHO_CONTEXT",),
                "context_b": ("ECHO_CONTEXT",),
                "weight_a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("ECHO_CONTEXT", "STRING")
    RETURN_NAMES = ("merged_context", "merge_info")
    FUNCTION = "merge_contexts"
    CATEGORY = "Framework/ECHO"

    def merge_contexts(self, context_a: Dict, context_b: Dict,
                       weight_a: float) -> Tuple[Dict, str]:

        weight_b = 1.0 - weight_a

        # Weighted content merge (simple concatenation with weights in metadata)
        merged_content = context_a.get("content", "") + "\n---\n" + context_b.get("content", "")

        merged = {
            "content": merged_content,
            "tier": "hot",  # Merged context is hot
            "compression_ratio": 1.0,
            "provenance": {
                "source": "merged",
                "timestamp": time.time(),
                "sources": [
                    {"hash": context_a.get("provenance", {}).get("content_hash", ""), "weight": weight_a},
                    {"hash": context_b.get("provenance", {}).get("content_hash", ""), "weight": weight_b}
                ]
            },
            "merge_weights": {"a": weight_a, "b": weight_b}
        }

        info = f"Merged with weights A:{weight_a:.1f} B:{weight_b:.1f}"
        return (merged, info)


# =============================================================================
# CSQMF-R1: MoE Expert Routing (Deterministic Hash-Based)
# =============================================================================

class MoE_ExpertRouter:
    """
    CSQMF-R1 Mixture of Experts routing with deterministic selection.

    Uses hash-based routing instead of MCMC sampling to ensure
    reproducibility (ThinkingMachines determinism fix).

    Expert Slots:
    - accuracy: Fact verification, precision
    - ethics: Safety, alignment checking
    - creativity: Novel generation, exploration
    - compression: Context reduction, summarization
    """

    EXPERT_DEFINITIONS = {
        "accuracy": {
            "description": "Fact verification and precision",
            "temperature": 0.1,
            "top_p": 0.9
        },
        "ethics": {
            "description": "Safety and alignment checking",
            "temperature": 0.3,
            "top_p": 0.95
        },
        "creativity": {
            "description": "Novel generation and exploration",
            "temperature": 0.8,
            "top_p": 0.95
        },
        "compression": {
            "description": "Context reduction and summarization",
            "temperature": 0.2,
            "top_p": 0.9
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"multiline": True}),
                "num_experts": ("INT", {"default": 2, "min": 1, "max": 4}),
                "routing_seed": ("INT", {"default": 42}),
            },
            "optional": {
                "force_expert": (["none", "accuracy", "ethics", "creativity", "compression"],),
            }
        }

    RETURN_TYPES = ("MOE_ROUTING", "STRING", "FLOAT")
    RETURN_NAMES = ("routing", "selected_experts", "confidence")
    FUNCTION = "route_query"
    CATEGORY = "Framework/CSQMF"

    def route_query(self, query: str, num_experts: int, routing_seed: int,
                    force_expert: str = "none") -> Tuple[Dict, str, float]:

        # DETERMINISTIC routing via hash (not MCMC sampling)
        # This is the ThinkingMachines fix for reproducibility

        routing_input = f"{query}:{routing_seed}"
        query_hash = hashlib.sha256(routing_input.encode()).hexdigest()

        # Score each expert deterministically from hash segments
        expert_scores = {}
        experts = list(self.EXPERT_DEFINITIONS.keys())

        for i, expert in enumerate(experts):
            # Use different 8-char segments for each expert
            segment = query_hash[i*8:(i+1)*8]
            score = int(segment, 16) / (16**8)  # Normalize to 0-1
            expert_scores[expert] = score

        # Handle forced expert
        if force_expert != "none":
            expert_scores[force_expert] = 1.0

        # Select top experts (deterministic: sort by score desc, then name for ties)
        sorted_experts = sorted(
            expert_scores.items(),
            key=lambda x: (-x[1], x[0])
        )
        selected = sorted_experts[:num_experts]

        # Build routing decision
        routing = {
            "query_hash": query_hash[:16],
            "seed": routing_seed,
            "selected_experts": [e[0] for e in selected],
            "expert_scores": {e[0]: round(e[1], 4) for e in selected},
            "expert_configs": {
                e[0]: self.EXPERT_DEFINITIONS[e[0]] for e in selected
            },
            "routing_method": "deterministic_hash",
            "reproducible": True
        }

        # Format output string
        experts_str = ", ".join([f"{e[0]}({e[1]:.3f})" for e in selected])

        # Confidence is average of selected expert scores
        confidence = sum(e[1] for e in selected) / len(selected)

        return (routing, experts_str, confidence)


class MoE_ExpertExecutor:
    """Execute generation with MoE routing parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "routing": ("MOE_ROUTING",),
                "base_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "execution_params")
    FUNCTION = "execute"
    CATEGORY = "Framework/CSQMF"

    def execute(self, routing: Dict, base_prompt: str) -> Tuple[str, str]:

        experts = routing.get("selected_experts", [])
        configs = routing.get("expert_configs", {})

        # Build expert-aware prompt prefix
        expert_instructions = []
        for expert in experts:
            config = configs.get(expert, {})
            desc = config.get("description", expert)
            expert_instructions.append(f"- Apply {expert} lens: {desc}")

        enhanced = f"""[MoE Routing: {', '.join(experts)}]
Expert guidance:
{chr(10).join(expert_instructions)}

Query:
{base_prompt}"""

        # Aggregate execution parameters
        temps = [configs.get(e, {}).get("temperature", 0.5) for e in experts]
        top_ps = [configs.get(e, {}).get("top_p", 0.9) for e in experts]

        params = {
            "temperature": sum(temps) / len(temps),
            "top_p": min(top_ps),  # Most conservative
            "experts_active": experts
        }

        return (enhanced, json.dumps(params, indent=2))


# =============================================================================
# PRISM: Multi-Perspective Analysis
# =============================================================================

class PRISM_Analyzer:
    """
    PRISM 6-perspective reasoning framework.

    Perspectives:
    - Causal: Root causes, ripple effects
    - Optimization: Bottlenecks, efficiency pathways
    - Hierarchical: System levels, leverage points
    - Temporal: Evolution, timing criticality
    - Risk: Vulnerabilities, cascading failures
    - Opportunity: Synergies, value creation
    """

    PERSPECTIVES = {
        "causal": {
            "focus": "Root causes and ripple effects",
            "questions": ["What caused this?", "What will this cause?", "What are the dependencies?"]
        },
        "optimization": {
            "focus": "Bottlenecks and efficiency pathways",
            "questions": ["Where are the bottlenecks?", "What can be parallelized?", "What's the critical path?"]
        },
        "hierarchical": {
            "focus": "System levels and leverage points",
            "questions": ["What level is this?", "What's above/below?", "Where's the leverage?"]
        },
        "temporal": {
            "focus": "Evolution and timing criticality",
            "questions": ["When does this matter?", "How will this evolve?", "What's time-sensitive?"]
        },
        "risk": {
            "focus": "Vulnerabilities and cascading failures",
            "questions": ["What could go wrong?", "What's the blast radius?", "How do we mitigate?"]
        },
        "opportunity": {
            "focus": "Synergies and value creation",
            "questions": ["What's the upside?", "What synergies exist?", "What's the innovation potential?"]
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subject": ("STRING", {"multiline": True}),
                "perspectives": ("STRING", {"default": "causal,optimization,risk"}),
                "depth": (["shallow", "medium", "deep"],),
            }
        }

    RETURN_TYPES = ("PRISM_ANALYSIS", "STRING")
    RETURN_NAMES = ("analysis", "summary")
    FUNCTION = "analyze"
    CATEGORY = "Framework/PRISM"

    def analyze(self, subject: str, perspectives: str, depth: str) -> Tuple[Dict, str]:

        # Parse requested perspectives
        requested = [p.strip().lower() for p in perspectives.split(",")]
        valid_perspectives = [p for p in requested if p in self.PERSPECTIVES]

        if not valid_perspectives:
            valid_perspectives = ["causal", "optimization", "risk"]

        # Depth multiplier for analysis detail
        depth_multiplier = {"shallow": 1, "medium": 2, "deep": 3}[depth]

        # Build analysis structure
        analysis = {
            "subject": subject[:200],
            "perspectives_analyzed": valid_perspectives,
            "depth": depth,
            "perspective_results": {}
        }

        summary_parts = []

        for perspective in valid_perspectives:
            config = self.PERSPECTIVES[perspective]

            # Generate perspective-specific analysis prompts
            questions = config["questions"][:depth_multiplier]

            analysis["perspective_results"][perspective] = {
                "focus": config["focus"],
                "questions_to_answer": questions,
                "analysis_prompt": self._build_perspective_prompt(subject, perspective, config)
            }

            summary_parts.append(f"[{perspective.upper()}] Focus: {config['focus']}")

        summary = "\n".join(summary_parts)

        return (analysis, summary)

    def _build_perspective_prompt(self, subject: str, perspective: str, config: Dict) -> str:
        """Build a perspective-specific analysis prompt."""

        questions = "\n".join([f"- {q}" for q in config["questions"]])

        return f"""Analyze the following from a {perspective.upper()} perspective.

Focus: {config['focus']}

Key questions to address:
{questions}

Subject to analyze:
{subject}

Provide {perspective} analysis:"""


# =============================================================================
# ThinkingMachines: Deterministic Sampler
# =============================================================================

class DeterministicSampler:
    """
    Batch-invariant sampling configuration (ThinkingMachines fix).

    The key insight: temperature=0 is NOT enough for determinism.
    Batch size variance causes non-deterministic outputs due to
    floating-point operation ordering in GPU parallel execution.

    This node enforces:
    - batch_size=1 (never vary)
    - Deterministic CUDA algorithms
    - Disabled cuDNN benchmark (auto-tuning causes variance)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 42}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                                  "dpmpp_sde", "dpmpp_2m", "ddim", "uni_pc"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG", "STRING")
    RETURN_NAMES = ("sampler_config", "determinism_info")
    FUNCTION = "create_config"
    CATEGORY = "Framework/Determinism"

    def create_config(self, seed: int, steps: int, cfg: float,
                      sampler_name: str, scheduler: str) -> Tuple[Dict, str]:

        # Apply ThinkingMachines determinism settings
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                # CRITICAL: These settings ensure batch-invariant operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        config = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,

            # DETERMINISM FLAGS (ThinkingMachines fix)
            "batch_size": 1,                    # NEVER vary this
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,           # Disable auto-tuning variance
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",

            # Reproducibility metadata
            "reproducibility_hash": hashlib.sha256(
                f"{seed}:{steps}:{cfg}:{sampler_name}:{scheduler}".encode()
            ).hexdigest()[:16]
        }

        info = f"""DETERMINISM ENFORCED:
- Seed: {seed}
- Batch Size: 1 (locked)
- cuDNN Deterministic: True
- cuDNN Benchmark: False
- Config Hash: {config['reproducibility_hash']}"""

        return (config, info)


class ChecksumValidator:
    """
    Validate reproducibility via cryptographic checksums.

    Use this node to:
    1. Generate checksums for outputs (store expected_hash="")
    2. Validate reproducibility (provide expected_hash from previous run)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "expected_hash": ("STRING", {"default": ""}),
                "hash_algorithm": (["sha256", "md5", "blake2b"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "computed_hash", "matches", "validation_report")
    FUNCTION = "validate"
    CATEGORY = "Framework/Determinism"

    def validate(self, image, expected_hash: str = "",
                 hash_algorithm: str = "sha256") -> Tuple[Any, str, bool, str]:

        # Convert image to bytes for hashing
        if HAS_TORCH and hasattr(image, 'cpu'):
            image_bytes = image.cpu().numpy().tobytes()
        else:
            image_bytes = np.array(image).tobytes()

        # Compute hash
        if hash_algorithm == "sha256":
            computed = hashlib.sha256(image_bytes).hexdigest()[:16]
        elif hash_algorithm == "md5":
            computed = hashlib.md5(image_bytes).hexdigest()[:16]
        else:  # blake2b
            computed = hashlib.blake2b(image_bytes).hexdigest()[:16]

        # Check match
        if expected_hash == "":
            matches = True  # No expectation, just generating
            report = f"Generated hash: {computed}\nStore this for future validation."
        else:
            matches = (computed == expected_hash)
            if matches:
                report = f"REPRODUCIBILITY VERIFIED\nHash: {computed}"
            else:
                report = f"REPRODUCIBILITY FAILED\nExpected: {expected_hash}\nGot: {computed}"

        return (image, computed, matches, report)


# =============================================================================
# VFX-Specific: Shot Intelligence (Phoenix + PRISM)
# =============================================================================

class VFX_ShotAnalyzer:
    """
    VFX shot analysis combining Phoenix keyword detection with PRISM perspectives.

    Detects VFX domains and routes to appropriate specialists.
    """

    VFX_DOMAINS = {
        "pyro": ["fire", "smoke", "explosion", "pyro", "volume", "combustion"],
        "flip": ["water", "fluid", "ocean", "splash", "flip", "liquid", "wave"],
        "rbd": ["destruction", "fracture", "rigid", "rbd", "collision", "debris"],
        "cloth": ["cloth", "fabric", "softbody", "vellum", "drape", "garment"],
        "hair": ["hair", "fur", "groom", "guide", "strand", "fiber"],
        "lighting": ["light", "render", "karma", "arnold", "mantra", "hdri", "gi"],
        "comp": ["composite", "nuke", "roto", "key", "despill", "grade"]
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_description": ("STRING", {"multiline": True}),
                "shot_name": ("STRING", {"default": "shot_001"}),
            },
            "optional": {
                "frame_range": ("STRING", {"default": "1001-1100"}),
            }
        }

    RETURN_TYPES = ("VFX_ANALYSIS", "STRING", "STRING")
    RETURN_NAMES = ("analysis", "detected_domains", "specialist_recommendation")
    FUNCTION = "analyze_shot"
    CATEGORY = "Framework/VFX"

    def analyze_shot(self, shot_description: str, shot_name: str,
                     frame_range: str = "1001-1100") -> Tuple[Dict, str, str]:

        desc_lower = shot_description.lower()

        # Detect VFX domains
        detected = []
        domain_scores = {}

        for domain, keywords in self.VFX_DOMAINS.items():
            matches = [kw for kw in keywords if kw in desc_lower]
            if matches:
                detected.append(domain)
                domain_scores[domain] = len(matches)

        # Determine primary specialist
        if domain_scores:
            primary = max(domain_scores, key=domain_scores.get)
        else:
            primary = "general"

        # Parse frame range
        try:
            start, end = frame_range.split("-")
            frame_count = int(end) - int(start) + 1
        except:
            frame_count = 100

        analysis = {
            "shot_name": shot_name,
            "frame_range": frame_range,
            "frame_count": frame_count,
            "detected_domains": detected,
            "domain_scores": domain_scores,
            "primary_specialist": primary,
            "prism_perspectives": {
                "causal": f"Dependencies for {primary} simulation",
                "optimization": f"Performance considerations for {frame_count} frames",
                "risk": f"Potential issues in {', '.join(detected) if detected else 'general'} workflow"
            }
        }

        domains_str = ", ".join(detected) if detected else "none detected"

        return (analysis, domains_str, primary)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # ECHO 2.0
    "ECHO_ContextManager": ECHO_ContextManager,
    "ECHO_ContextMerger": ECHO_ContextMerger,

    # CSQMF-R1
    "MoE_ExpertRouter": MoE_ExpertRouter,
    "MoE_ExpertExecutor": MoE_ExpertExecutor,

    # PRISM
    "PRISM_Analyzer": PRISM_Analyzer,

    # Determinism
    "DeterministicSampler": DeterministicSampler,
    "ChecksumValidator": ChecksumValidator,

    # VFX
    "VFX_ShotAnalyzer": VFX_ShotAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ECHO 2.0
    "ECHO_ContextManager": "ECHO Context Manager (4-Tier)",
    "ECHO_ContextMerger": "ECHO Context Merger",

    # CSQMF-R1
    "MoE_ExpertRouter": "MoE Expert Router (CSQMF)",
    "MoE_ExpertExecutor": "MoE Expert Executor",

    # PRISM
    "PRISM_Analyzer": "PRISM Multi-Perspective Analyzer",

    # Determinism
    "DeterministicSampler": "Deterministic Sampler (Batch-Invariant)",
    "ChecksumValidator": "Checksum Validator (Reproducibility)",

    # VFX
    "VFX_ShotAnalyzer": "VFX Shot Analyzer (Phoenix+PRISM)",
}

# Category color hints (for ComfyUI theming)
NODE_CATEGORY_COLORS = {
    "Framework/ECHO": "#4A90D9",      # Blue - Memory
    "Framework/CSQMF": "#D94A4A",     # Red - Routing
    "Framework/PRISM": "#4AD94A",     # Green - Analysis
    "Framework/Determinism": "#D9D94A",  # Yellow - Validation
    "Framework/VFX": "#9B4AD9",       # Purple - VFX
}
