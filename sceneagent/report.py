"""HTML experiment view for the sister grid.

The page follows the shape of SceneAgent figure 14 and the website's
place-success matrix: one row per object sister, one column per lighting
name, with successes over n and the 95% Wilson interval. It states that the
numbers come from the scripted geometric check.
"""

from __future__ import annotations

import html


def _pct(rate):
    return f"{100.0 * rate:.0f}%"


def render_report(scene_name, grid):
    header = "".join(f"<th>{html.escape(name)}</th>" for name in grid["lighting"])
    body = []
    for row in grid["rows"]:
        cells = []
        for name in grid["lighting"]:
            cell = row["cells"][name]
            low, high = cell["wilson95"]
            cells.append(
                "<td>"
                f"<div class='rate'>{_pct(cell['rate'])}</div>"
                f"<div class='count'>{cell['successes']} / {cell['trials']}</div>"
                f"<div class='wilson'>Wilson 95% {100 * low:.1f}% to {100 * high:.1f}%</div>"
                "<div class='note'>Lighting not simulated</div>"
                "</td>"
            )
        reasons = ", ".join(row["reasons"]) or "accepted"
        body.append(
            "<tr>"
            f"<th>{html.escape(row['sister'])}<div class='why'>{html.escape(reasons)}</div></th>"
            + "".join(cells)
            + "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(scene_name)} sister grid</title>
<style>
  body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; background: #f7f4ef; }}
  h1 {{ font-weight: 500; }}
  p {{ max-width: 42rem; line-height: 1.45; }}
  table {{ border-collapse: collapse; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 0.6rem; vertical-align: top; text-align: left; }}
  th {{ background: #f0ebe3; }}
  .rate {{ font-size: 1.3rem; }}
  .count, .wilson, .note, .why {{ color: #555; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>{html.escape(scene_name)}</h1>
<p>Place-inside success for each object sister. The check is the phase-1 scripted geometric expert: the sister must pass the mesh gates, and the start pose must lie inside the reach disc. This is not a pi0.5 rollout and it does not use the success counts published on the SceneAgent website.</p>
<p>Lighting column names match the website grid. The geometric check does not depend on light, so the columns repeat and each cell says lighting was not simulated.</p>
<table>
<thead><tr><th>Object sister</th>{header}</tr></thead>
<tbody>
{''.join(body)}
</tbody>
</table>
</body>
</html>
"""
