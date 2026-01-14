# ComfyUI Framework Ecosystem
# Copyright (C) 2025 Joseph Ibrahim <joseph@josephibrahim.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing available: See COMMERCIAL_LICENSE.md

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
