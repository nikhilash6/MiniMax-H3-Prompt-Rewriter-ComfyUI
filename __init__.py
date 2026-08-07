"""ComfyUI entry point for the MiniMax-H3 T2VA prompt rewriter."""

from .minimax_h3_rewriter import routes as _routes  # registers HTTP routes
from .minimax_h3_rewriter.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
