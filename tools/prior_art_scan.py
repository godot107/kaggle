#!/usr/bin/env python
"""Prior-art scan for a Kaggle project: survey existing community work before building.

Lists the top community notebooks for a dataset (or competition) ranked by votes and
by discussion, and optionally pulls the highest-signal ones so you can read their code.
Writes a markdown report you can drop into the project folder and work against.

Why: for the *same dataset*, the community has usually already mapped the obvious
approaches, the common mistakes, and (in the comments) the sharpest critiques. Reading
that first tells you the baseline to beat and where your differentiated angle should be.

Usage:
    python tools/prior_art_scan.py --dataset suddharshan/retail-price-optimization \
        --out retail-pricing/PRIOR_ART.md --pull 3
    python tools/prior_art_scan.py --competition titanic --top 15

Requires kaggle auth (~/.kaggle/access_token or kaggle.json). The notebook *comments*
themselves aren't in the Kaggle API -- open the top refs' web pages to read those
(this report links them for you).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _scan(api, *, dataset=None, competition=None, sort_by="voteCount", top=15):
    """Return up to `top` kernels for the dataset/competition, by sort_by."""
    return api.kernels_list(
        dataset=dataset or "",
        competition=competition or "",
        sort_by=sort_by,
        page_size=top,
    )


def _rows(kernels):
    out = []
    for k in kernels:
        out.append({
            "ref": getattr(k, "ref", ""),
            "title": (getattr(k, "title", "") or "").strip(),
            "author": getattr(k, "author", ""),
            "votes": getattr(k, "total_votes", ""),
            "lastRun": str(getattr(k, "last_run_time", "") or "")[:10],
        })
    return out


def _table(rows):
    if not rows:
        return "_none found_\n"
    lines = ["| votes | title | author | ref (open for code + comments) | last run |",
             "|---|---|---|---|---|"]
    for r in rows:
        url = f"https://www.kaggle.com/code/{r['ref']}"
        lines.append(f"| {r['votes']} | {r['title']} | {r['author']} | [{r['ref']}]({url}) | {r['lastRun']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="dataset slug, e.g. user/dataset-slug")
    g.add_argument("--competition", help="competition slug")
    ap.add_argument("--top", type=int, default=15, help="how many to list (default 15)")
    ap.add_argument("--pull", type=int, default=0, help="pull the top-N notebooks' source to read")
    ap.add_argument("--pull-dir", default="", help="where to pull (default: <out dir>/_prior_art)")
    ap.add_argument("--out", default="", help="write the markdown report here")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    target = args.dataset or args.competition
    kind = "dataset" if args.dataset else "competition"
    kw = {"dataset": args.dataset} if args.dataset else {"competition": args.competition}

    by_votes = _rows(_scan(api, **kw, sort_by="voteCount", top=args.top))
    by_comments = _rows(_scan(api, **kw, sort_by="commentCount", top=args.top))

    report = (
        f"# Prior-art scan — {kind} `{target}`\n\n"
        "Existing community notebooks for the same data. Read the top few before building:\n"
        "their consensus approach is the baseline to beat, and the most-commented ones are\n"
        "where the sharp critiques live (open the ref to read comments — not in the API).\n\n"
        "## Highest-rated (by votes)\n" + _table(by_votes) +
        "\n## Most-discussed (by comment count — look here for constructive criticism)\n"
        + _table(by_comments) +
        "\n## Synthesize (fill in after reading)\n"
        "- **Consensus approach / baseline:** \n"
        "- **Common mistakes the comments call out:** \n"
        "- **What to adopt:** \n"
        "- **My differentiated angle (where I go beyond the crowd):** \n"
        "- **Credit / license notes:** \n"
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"wrote {args.out}  ({len(by_votes)} by votes, {len(by_comments)} by comments)")
    else:
        print(report)

    if args.pull:
        pull_dir = Path(args.pull_dir or (Path(args.out).parent if args.out else ".") / "_prior_art")
        pull_dir.mkdir(parents=True, exist_ok=True)
        for r in by_votes[: args.pull]:
            dest = pull_dir / r["ref"].replace("/", "__")
            dest.mkdir(exist_ok=True)
            subprocess.run(["kaggle", "kernels", "pull", r["ref"], "-p", str(dest), "-m"], check=False)
        print(f"pulled top {args.pull} notebooks to {pull_dir}/ — read them, then fill in the synthesis")


if __name__ == "__main__":
    main()
