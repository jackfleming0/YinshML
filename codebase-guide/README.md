# YinshML — Guided Tour of the Codebase

An interactive, single-page HTML course that walks a developer new to this repo
through how YinshML works, end to end. No build step or server needed to view it.

## View it

Open **`index.html`** in any browser:

```bash
open codebase-guide/index.html      # macOS
xdg-open codebase-guide/index.html  # Linux
```

It's fully self-contained (one HTML file + `styles.css` + `main.js`). Fonts load
from Google Fonts, so the first load looks best with a network connection.

## What's inside

Six scroll-through modules, each with code↔plain-English translations, animated
data-flow diagrams, quizzes, and hover-glossary tooltips on the domain jargon:

1. **What YinshML Actually Does** — the AlphaZero-style self-play → train → evaluate loop
2. **The Board and the Rules** — the `yinsh_ml/game/` engine, hex geometry, phases
3. **Teaching the Computer to See the Board** — state encoding + the neural network
4. **How the AI Thinks** — MCTS, negamax/alpha-beta, heuristics, transposition tables
5. **Learning From Experience** — self-play data, the two-headed loss, Parquet storage
6. **Orchestration, Evaluation & Finding Your Way** — the supervisor, ELO, and repo map

## Rebuild

The page is assembled from parts (so modules stay editable). To regenerate
`index.html` after editing anything in `modules/` or `_base.html`:

```bash
cd codebase-guide && bash build.sh
```

## How it was made

Generated with the [`codebase-to-course`](https://github.com/zarazhangrui/codebase-to-course)
Claude Code skill, with content adapted for a technical (developer) audience rather
than the skill's default non-technical framing. The `styles.css` / `main.js` design
assets are from that skill; the module content is specific to this repo.
