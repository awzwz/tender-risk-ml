"""Builds the Assignment 4 notebooks from Python: markdown in English and Russian, code shared.

    nb = Notebook()
    nb.md(en, ru)      # markdown cell (the Russian text goes to the reading copy)
    nb.code(src)       # code cell
    nb.save("P1_data_preparation.ipynb")             # English notebook (executed afterwards with nbconvert)
    nb.ru_copy("P1_data_preparation.ipynb", "P1_data_preparation_ru.ipynb")   # executed EN -> RU, outputs kept

`refresh_markdown` rewrites the markdown of an already executed notebook (for conclusions written after a run)
without re-executing it; code cells must be unchanged.
"""
import nbformat
from textwrap import dedent

KERNEL = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}


class Notebook:
    def __init__(self):
        self.cells = []          # ("md", en, ru) or ("code", src)

    def md(self, en, ru):
        self.cells.append(("md", dedent(en).strip(), dedent(ru).strip()))

    def code(self, src):
        self.cells.append(("code", dedent(src).strip()))

    def _nb(self, lang="en"):
        nb = nbformat.v4.new_notebook()
        nb.metadata = KERNEL
        for c in self.cells:
            if c[0] == "md":
                nb.cells.append(nbformat.v4.new_markdown_cell(c[1] if lang == "en" else c[2]))
            else:
                nb.cells.append(nbformat.v4.new_code_cell(c[1]))
        return nb

    def save(self, path):
        nbformat.write(self._nb("en"), path)

    def _swap(self, src_path, dst_path, lang):
        nb = nbformat.read(src_path, as_version=4)
        texts = [c[1] if lang == "en" else c[2] for c in self.cells if c[0] == "md"]
        codes = [c[1] for c in self.cells if c[0] == "code"]
        mds = [c for c in nb.cells if c.cell_type == "markdown"]
        cds = [c for c in nb.cells if c.cell_type == "code"]
        assert len(mds) == len(texts) and len(cds) == len(codes), "builder and notebook differ in structure"
        assert all(c.source == s for c, s in zip(cds, codes)), "code changed since the notebook was executed"
        for c, t in zip(mds, texts):
            c.source = t
        nbformat.write(nb, dst_path)

    def ru_copy(self, executed_en, ru_path):
        self._swap(executed_en, ru_path, "ru")

    def refresh_markdown(self, executed_en):
        self._swap(executed_en, executed_en, "en")
