# visualization.py
# ─────────────────
# Loads visualisation.py (British spelling) by file path so Python never
# tries to resolve it as a dotted module path (which breaks when the
# project root is on sys.path under a deeply-nested directory).

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_VIS_PATH = os.path.join(_HERE, "visualisation.py")

_spec = importlib.util.spec_from_file_location("visualisation", _VIS_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export everything pipeline.py needs
plot_coherence_comparison = _mod.plot_coherence_comparison
plot_diversity            = _mod.plot_diversity
plot_error_propagation    = _mod.plot_error_propagation
save_pyldavis             = _mod.save_pyldavis
save_bertopic_visuals     = _mod.save_bertopic_visuals
plot_word_clouds          = _mod.plot_word_clouds
plot_doc_topic_heatmap    = _mod.plot_doc_topic_heatmap
print_summary_table       = _mod.print_summary_table
ENGINE_COLORS             = _mod.ENGINE_COLORS
METHOD_PATTERNS           = _mod.METHOD_PATTERNS
WORDCLOUD_COLORMAPS       = _mod.WORDCLOUD_COLORMAPS
PALETTE                   = getattr(_mod, "PALETTE", _mod.WORDCLOUD_COLORMAPS)
