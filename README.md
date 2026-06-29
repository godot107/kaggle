# kaggle-projects

A home for self-contained Kaggle / public-dataset projects. Each lives in its own
subdirectory with its own `requirements.txt`, data folder (gitignored), and notebooks.

## Projects

| Project | Status | Summary |
|---|---|---|
| [retail-pricing](retail-pricing/) | 🟡 in progress | Retail price optimization via causal price-elasticity → margin. |

## Conventions

- **Data is never committed.** Each project has `data/raw/` (gitignored); document how to
  fetch it in that project's README.
- One `requirements.txt` per project; install into the shared workspace `.venv` or a
  project-local one.
- Notebooks live in `notebooks/`; reusable code in `src/`.
