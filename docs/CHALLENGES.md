# Challenges

Real obstacles hit and how they were attacked — especially free-hardware
workarounds. This is the honest engineering story recruiters remember.

> Update whenever you hit a wall and resolve (or work around) it.

## Template

```
### <Challenge title>   (YYYY-MM-DD, Phase N)
- **Problem:** what blocked progress.
- **Constraint:** hardware / free-tier / data limit that made it hard.
- **Attempts:** what was tried (incl. what failed).
- **Resolution:** what worked, and the tradeoff accepted.
```

---

_Anticipated (from design risk analysis):_
- Free-text ingredient parsing messiness (Phase 1).
- Substitution eval ground-truth sparsity (Phase 2).
- Colab/Kaggle session timeouts + GPU quota during training.
- 4 GB VRAM (MX150) → inference-only locally, ONNX export from cloud training.
