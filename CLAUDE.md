# kaggle-projects — Claude notes

## What this is
A standalone git repo (separate from the workspace monorepo) holding Kaggle / public-dataset
portfolio projects. Intended to be pushed to GitHub as its own repo.

## Structure
```
kaggle-projects/
├── retail-pricing/      # first project (see its README + CLAUDE.md)
└── <future projects>/
```

## Per-project layout
```
<project>/
├── data/raw/            # gitignored — fetch instructions in the project README
├── notebooks/           # exploration, numbered (01_eda.ipynb, ...)
├── src/                 # reusable code
├── requirements.txt
└── README.md
```

## Conventions / decisions
- **Never commit data.** `data/` is gitignored except `.gitkeep`.
- Honest, leakage-free evaluation over leaderboard chasing — these are portfolio pieces,
  the writeup/README narrative matters as much as the metric.
- Prefer interpretable, defensible methods; only reach for deep learning when it's justified.
