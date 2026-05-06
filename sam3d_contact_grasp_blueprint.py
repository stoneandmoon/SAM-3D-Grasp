#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sam3d_contact_grasp_blueprint.py

把 SAM3DContactGraspSkill + McpServer 组合成 DimOS 可运行 blueprint。
"""

from dimos.core.blueprints import autoconnect
from dimos.agents.mcp.mcp_server import McpServer

from dimos_sam3d_grasp_skill import SAM3DContactGraspSkill


sam3d_contact_grasp_mcp = autoconnect(
    SAM3DContactGraspSkill.blueprint(),
    McpServer.blueprint(),
)
