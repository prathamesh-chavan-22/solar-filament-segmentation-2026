"""Convert a `# %%` percent-format script into a Jupyter notebook.

Usage: python tools/py2nb.py src/notebook.py notebooks/solar-filament-yolo-seg.ipynb
"""
import json
import re
import sys


def convert(src, dst):
    text = open(src, encoding="utf-8").read()
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if "[markdown]" in header:
            source = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body})
    for c in cells:
        c["source"] = c["source"].splitlines(keepends=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
