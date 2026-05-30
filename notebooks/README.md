# Notebooks

**Thin only.** Notebooks install the package and call it — no business logic in
cells. This keeps logic tested, reusable, and identical across local / Colab /
Kaggle.

Top cell of every notebook:

```python
!pip install -q "git+https://github.com/<you>/pantrychef.git@main"
# then import and call:
from pantrychef.ingredients import parse   # example (Phase 1)
```

Naming: `NN_phaseN_<purpose>.ipynb` (e.g. `01_phase1_parser_eval.ipynb`).

Heavy training runs here (Colab/Kaggle GPU); export artifacts (weights, ONNX)
for local inference on the MX150.
