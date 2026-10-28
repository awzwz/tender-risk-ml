"""
Russian reading copies of the executed A3 notebooks (not for submission).

The English notebook is the deliverable; the copy keeps its code and outputs unchanged (no re-execution, so no
model is refitted and no file is overwritten) and replaces every markdown cell with the Russian text from
notebooks/ru_a3/<nn>.py, in order. Chart labels stay in English; each chart is described in the Russian text.

Run:  python notebooks/make_ru_a3.py 04 05a 05b 06 07
"""
import importlib.util
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
NOTEBOOKS = {"04": "04_model_comparison_tuning", "05a": "05a_text_encoders_finetune", "05b": "05b_text_comparison",
             "06": "06_llm_qlora", "07": "07_results"}


def build(key: str) -> Path:
    src = HERE / f"{NOTEBOOKS[key]}.ipynb"
    spec = importlib.util.spec_from_file_location(f"ru_{key}", HERE / "ru_a3" / f"{key}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    nb = nbformat.read(src, as_version=4)
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    assert len(md_cells) == len(mod.RU), f"{key}: {len(md_cells)} markdown cells in the notebook, {len(mod.RU)} translations"
    for cell, text in zip(md_cells, mod.RU):
        cell.source = text.strip("\n")
    out = HERE / f"{NOTEBOOKS[key]}_ru.ipynb"
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for k in sys.argv[1:] or NOTEBOOKS:
        print("written", build(k))
