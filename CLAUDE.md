# kaggle-projects — Claude notes

## What this is
A standalone git repo (separate from the workspace monorepo) holding Kaggle / public-dataset
portfolio projects. Intended to be pushed to GitHub as its own repo.

## Structure
```
kaggle-projects/
├── tools/               # cross-project helpers (e.g. prior_art_scan.py)
├── retail-pricing/      # first project (see its README + CLAUDE.md)
└── <future projects>/
```

## Workflow: prior-art scan (do this FIRST on every new project)
Before building, survey existing community work on the same dataset/competition: it shows the
baseline to beat and where a differentiated angle lives. The crowd's *consensus* is rarely the
best take — find what they all skip.
```bash
python tools/prior_art_scan.py --dataset <user/slug> --out <project>/PRIOR_ART.md --pull 3
# or:  --competition <slug>
```
This writes a ranked report (by votes, and by comment count) and pulls the top notebooks' code.
Then:
1. **Read the top 2–3 notebooks' code** (pulled locally) — what's the consensus method?
2. **Read their comments in the browser** (Kaggle comments aren't in the API) — sharp critiques,
   leakage/target/cost objections live there.
3. **Fill in the PRIOR_ART.md synthesis**: consensus baseline, what to adopt, and **my edge**
   (where I deliberately go beyond the crowd). Adapt, don't copy; credit specific ideas; respect
   licenses. Worked example: `retail-pricing/PRIOR_ART.md` — the top voted notebooks predict
   price with RandomForest+SHAP; our causal-elasticity framing is the differentiator.

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
