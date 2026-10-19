from __future__ import annotations

import re

from scripts.report import sections
from scripts.stats import PROVENANCE, results_md


def test_sections_render_from_committed_ablations_without_broken_tables() -> None:
    secs = sections()
    assert {"oracle", "rank_corr", "overlap", "sgd", "beta", "dense", "cost"} <= set(secs)
    for md in secs.values():
        for line in md.splitlines():
            if line.startswith("|"):
                cells = [c for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
                assert all(c.strip() for c in cells), line


def test_results_md_leads_with_the_headline_and_ends_with_limitations() -> None:
    md = results_md([], [], "HEADLINE", {"cost": "COST"}, "OBS")
    assert md.index("## Headline claim") < md.index("HEADLINE") < md.index("## Cost evidence")
    assert md.rstrip().splitlines()[-1].startswith("- Eight seeds")
    assert "## Limitations" in md and "not** described as pre-registered" in PROVENANCE
