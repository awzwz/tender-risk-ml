"""Rewrites markdown of executed notebooks from their builders (no re-execution) and makes the Russian copies.
usage: python finalize.py p1 [p2 ...]"""
import importlib
import sys

NAMES = {"p1": "P1_data_preparation", "p2": "P2_baseline_metrics", "p3": "P3_model_comparison_tuning",
         "p4": "P4_transfer_learning", "p5": "P5_fine_tuning", "p6": "P6_llm_lora"}
for key in sys.argv[1:]:
    nb = importlib.import_module(f"build_{key}").nb
    name = NAMES[key]
    nb.refresh_markdown(f"{name}.ipynb")
    nb.ru_copy(f"{name}.ipynb", f"{name}_ru.ipynb")
    print("finalized", name)
