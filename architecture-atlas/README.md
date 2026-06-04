# YinshML — Internal Architecture Atlas

An interactive, single-page technical reference for engineers working **on**
YinshML (not learning what it does — see `../codebase-guide/` for that). It maps
how the subsystems fit together: the exact objects, fields, and shapes that cross
each boundary; the real function signatures at each seam; the call chains; and the
cross-file invariants that break the build if you touch one side.

## View it

Open **`index.html`** in any browser (self-contained: `index.html` + `styles.css`
+ `main.js`). Or open the portable single-file build:
**`yinshml-architecture-atlas-standalone.html`** (CSS/JS inlined — one shareable file).

```bash
open architecture-atlas/index.html
```

## Sections

1. **The System Map** — clickable subsystem diagram; the object each package owns
2. **The Shared Currency: Core Data Types** — `Position`/`Move`/`GameState`, the
   `(6,11,11)` tensor, `(move_probs[7433], value)`, the stored record — with
   producer→consumer annotations
3. **The State ⇄ Network Seam** — `encode_state` → `predict`/`predict_batch` →
   `decode_move`; the `use_enhanced_encoding` plumbing; the load-time hard contracts
4. **The Search ⇄ Evaluation Seam** — what `MCTS` is wired to, the batched leaf-eval
   route to `predict_batch`, the heuristic stack, shared TT/Zobrist infra
5. **The Self-Play ⇄ Storage ⇄ Training Seam** — `train_iteration()` call chain,
   the loss boundary, the move-selection policies, config plumbing
6. **Cross-Cutting Infrastructure & Invariants** — memory pools, the config→consumer
   map, and the six **change-one-change-all** tripwires (policy size 7433, channel
   count, Zobrist contents, A1 sentinel, hex `DIRECTIONS`, `value_mode`)

Every claim carries a real `file_path:line` reference (clickable in an editor).

## Rebuild / regenerate the standalone

```bash
cd architecture-atlas && bash build.sh         # reassemble index.html from modules/
# then regenerate the portable single-file copy:
python3 - <<'PY'
h=open('index.html').read();c=open('styles.css').read();j=open('main.js').read()
h=h.replace('<link rel="stylesheet" href="styles.css">','<style>\n'+c+'\n</style>')
h=h.replace('\n  <script src="main.js" defer></script>','')
h=h.replace('</body>','  <script>\n'+j+'\n  </script>\n</body>')
open('yinshml-architecture-atlas-standalone.html','w').write(h)
PY
```

Design assets (`styles.css`/`main.js`) are shared with `../codebase-guide/`, from
the [`codebase-to-course`](https://github.com/zarazhangrui/codebase-to-course) skill;
the content here is an internal-facing adaptation (signature↔contract blocks,
call-sequence animations, invariant tables — no quizzes/metaphors).
