"""Execute the supervised notebook in a real Jupyter kernel and save outputs."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path,
                        default=root / "ml/notebooks/02_supervised_comparison.ipynb")
    parser.add_argument("--kernel", default="harss")
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root)}})
    client.on_cell_executed = lambda cell, cell_index, **_: print(
        f"Executed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    client.execute()
    nbformat.write(notebook, args.notebook)
    print(f"Saved executed notebook: {args.notebook.name}")


if __name__ == "__main__":
    main()
