# Autonomous cloud experiments with Claude

How to use Claude Code to manage long-running ML experiments on a remote cloud box across hours-to-days, with periodic check-ins, while you're asleep / away. This document captures what worked across two 24-hour autonomous sessions on YinshML.

## TL;DR — the loop

1. **You** push code + a config to the branch on the laptop, then tell Claude what to investigate.
2. **Claude** SSHes to the cloud box, launches an experiment under `nohup`, schedules a self-perpetuating wakeup, and disconnects.
3. **Each wakeup** (typically 25–40 minutes apart), Claude reconnects, greps the log, parses results, decides whether to act, and re-schedules itself.
4. **State** is preserved across wakeups via files on disk: a status log in `~/.claude/projects/.../memory/`, the project's git tree, and result docs committed to the branch.
5. **Action** at wakeup boundaries: trigger the next experiment, kill a hung run, write a summary, etc. Claude's prompts to its future self carry just enough context to pick up where it left off.

You wake up to a final results doc and committed changes. The compute ran while you slept.

## Prerequisites

### 1. SSH config (laptop, `~/.ssh/config`)

Two aliases for the same host: one with port-forwarding, one without. The one without is for one-off commands that shouldn't compete for the local forward port.

```ssh-config
# YinshML cloud training box (Vast.ai 4090)
# Update HostName/Port if the instance is reprovisioned.

Host cloud
    HostName <ip>
    Port <port>
    User root
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host cloud-dash
    HostName <ip>
    Port <port>
    User root
    LocalForward 8080 localhost:8080
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

`ServerAliveInterval` keeps long-idle SSH calls from getting dropped by NAT timeouts.

### 2. Cloud box environment

The box should already have the repo cloned and a Python environment activated by default at login (e.g., `(main)` conda env at `/venv/main`). Use full paths inside scripts (`/venv/main/bin/python`) — non-login SSH shells don't auto-activate conda.

```bash
ssh cloud 'pwd && which python && git log --oneline -3'
```

If `import yinsh_ml` fails: `pip install -e .` once, then never again.

### 3. Per-instance state on cloud

Always written by Claude. You don't manage these:

- `/workspace/YinshML/<experiment>.log` — stdout from each experiment
- `/workspace/YinshML/<experiment>.pid` — saved PID for kill+probe
- `/workspace/YinshML/run_<experiment>.sh` — the bash wrapper Claude writes per experiment
- `runs_<config>/<timestamp>/...` — actual training artifacts (checkpoints, metrics, replay buffer)

## The four patterns that make this work

### Pattern 1: `nohup` + `disown` for survive-disconnect launches

The naive `ssh cloud "python long_thing.py &"` dies when SSH disconnects. The robust pattern:

```bash
ssh cloud 'cat > /workspace/YinshML/run_<x>.sh <<EOF
#!/bin/bash
cd /workspace/YinshML
LOG=/workspace/YinshML/<x>.log
echo "=== <x> starting at \$(date -u) ===" > \$LOG
/venv/main/bin/python scripts/run_training.py --config configs/<x>.yaml >> \$LOG 2>&1
# ... post-train evals ...
echo "=== ALL DONE at \$(date -u) ===" >> \$LOG
EOF
chmod +x /workspace/YinshML/run_<x>.sh
nohup /workspace/YinshML/run_<x>.sh > /dev/null 2>&1 &
disown
sleep 3
ps aux | grep -E "<x>|run_training" | grep -v grep | head'
```

Key elements:
- **Heredoc writes the script to disk** so it survives the parent SSH closing.
- **`nohup … &` + `disown`** detaches from the parent shell so a SIGHUP doesn't kill it.
- **`sleep 3` then `ps aux`** verifies the process actually started — without this you can't tell if the launch worked.
- **Log file is the durable state** — when Claude wakes up later, the only thing it can read is the log.

Single-line `ssh cloud "nohup … &"` patterns are flaky because of how shells propagate signals. Always use the heredoc-script-to-disk approach for anything > 5 minutes.

### Pattern 2: chained experiments in one script

When you want to run N experiments back to back without intervention, write the chain into the script and let Claude monitor at the chain boundary, not at each step:

```bash
run_one() {
  local LABEL=$1
  echo "=== $LABEL starting at $(date -u) ===" >> $LOG
  /venv/main/bin/python scripts/run_training.py --config configs/${LABEL}.yaml >> $LOG 2>&1
  RUN=$(ls -dt runs_${LABEL}/*/ | head -1)
  for i in 0 1 2 3 4; do
    EMA=${RUN}iteration_$i/checkpoint_iteration_${i}_ema.pt
    [ -f $EMA ] || continue
    /venv/main/bin/python scripts/eval_vs_heuristic.py --checkpoint $EMA … --label ${LABEL}_iter${i} >> $LOG 2>&1
  done
}

run_one config_a
run_one config_b
run_one config_c
echo "=== ALL DONE at $(date -u) ===" >> $LOG
```

Now there's a single `ALL DONE` marker that Claude greps for to know the chain completed. No need to launch each experiment manually with N round-trips.

### Pattern 3: self-perpetuating wakeups with explicit context

The `ScheduleWakeup` tool fires a future "user message" with a prompt you write. The trick is to make that prompt **self-contained enough that Claude on wakeup, with no fresh memory, can pick up correctly**. A working template:

```
[autonomous monitoring — <experiment name>] <one-sentence what's running>.
PID <N>. Log: <path>. ETA done <UTC time>.
Each wakeup: ssh cloud, grep `<key markers>` from <log path>, append to <status doc path>.
Re-schedule unless ALL DONE.
<key context: prior results, what the next decision criterion is>.
```

The prompt must include:
- **What's running and how to find it** (PID, log path).
- **The ETA** — so Claude knows whether to expect results yet.
- **The grep command** — exact patterns to look for.
- **Re-schedule policy** — explicit conditions for stopping.
- **Decision criteria** — e.g., "if metric > X, launch follow-up; if < X, write up and stop."
- **Prior numbers** — comparison targets so Claude doesn't have to guess what "good" means.

Avoid relying on Claude's conversational memory across wakeups. Treat each wake-up as a cold start that's been handed a stubby briefing.

#### Picking the wake-up interval

- **270 seconds (~4.5 min)**: cache-warm, cheap. Use when you're polling for a state change you expect imminently.
- **1500–1800 seconds (25–30 min)**: cache-cold, normal. Use for "check on training progress." Most experiments fall here.
- **2400 seconds (40 min)**: longer-running iters. Use when iter cadence is ~50 min/iter and there's nothing in between.

Don't pick 300 seconds — worst-of-both: you pay the cache miss without amortizing it. Either go shorter (270) or longer (1500+).

### Pattern 4: status-log-as-source-of-truth

Across many wakeups, conversational context fades. Two durable artifacts keep things coherent:

**A. Append-only status log**
Lives at `~/.claude/projects/<project>/memory/<experiment>_watch_log.md`. Claude appends a few lines at every wakeup:

```
### t=140min (14:32 UTC, wakeup #5)
- PID alive, 23min/iter pace
- iter 3 ✓: pol=3.78, val=1.61, ELO=1494
- Per-iter anchor n=4: 0/4 (still looking for spike)
- Re-scheduling for 15:02 UTC
```

When you wake up, you can read this top-down for moment-by-moment trajectory.

**B. Result doc on the git branch**
Lives in the repo, e.g., `EXPERIMENT_RESULTS_24H.md`. Claude commits + pushes after every meaningful result. This is the "executive summary" that survives even if the watch log gets long.

The split: **status log = trajectory** (how we got here), **result doc = conclusions** (what we know now).

## The actual session loop, step by step

```
You:    "I want to test hypothesis X. Set it up."
Claude: writes config, writes run script, launches, schedules wakeup +30min
        [you go to bed]

[+30 min, wakeup]
Claude: SSHes, greps log, reads "iter 0 complete in 14 min"
        appends to watch log, schedules next wakeup
        decides: keep going, no action needed

[+60 min]
Claude: SSHes, sees iter 2 complete + per-iter anchor n=4=4/4
        notes promising signal
        appends to watch log
        re-schedules

[+90 min, training finished, eval running]
Claude: SSHes, sees n=60 eval result = 60/60
        BIG result; commits findings to branch
        decides: queue follow-up experiment because main hypothesis confirmed
        writes new config, launches, schedules wakeup +40min

[continues for hours]

[you wake up]
You:    cat EXPERIMENT_RESULTS_24H.md
        cat ~/.claude/projects/.../memory/<experiment>_watch_log.md
        decide what's next
```

## Common failure modes and how to avoid them

### SSH dropouts

The cloud box may transiently lose connectivity. If Claude ssh's during a dropout, the command times out. Wakeup loop just retries on the next interval; usually the box is back. If unreachable for >30 min, treat the instance as gone — write up findings with what you have. (This actually happened mid-session; the run kept going on the cloud box, but Claude lost ability to monitor for ~10 min.)

### Hung evals (`time-limit-per-move 0`)

Eval against `HeuristicAgent(depth=3)` without a per-move time cap can stall on certain board positions due to alpha-beta blowup. Always pass `--time-limit-per-move 5.0` (or 10.0) for depth-3+ evals. You'll get correct results within a tractable wall time.

### "200 sims is feasible at 200 games" — no it isn't

Sim cost is roughly O(sims × moves). 48 sims at 200 games = 17 min/iter. 200 sims at 200 games = 13 hours/iter. Always sanity-check wall time before launching multi-iter chains — `200 games × 96 moves × Y_seconds_per_sim_avg` should be your back-of-envelope.

### Stale wakeup prompts

If you re-schedule frequently with new context, sometimes an older wakeup fires after a newer one. Claude sees a stale prompt that conflicts with current state. Mitigation: each wakeup grep the log directly to read actual state instead of trusting the prompt — the prompt is just a hint about what to look for, not the source of truth.

### Process IDs go stale

`<experiment>.pid` files refer to a specific PID. If the process exits (crashes, finishes) and a new process gets the same PID, `kill $(cat .pid)` could kill the wrong thing. Always pair `ps -p <pid> -o command` to verify it's still the experiment before acting.

### Disk fills up

Each iteration writes a checkpoint (~137 MB live + ~137 MB EMA). 25 iterations = ~7 GB per run. Several runs add up fast on a 50-GB cloud disk. Periodically:

```bash
ssh cloud 'du -sh runs_*'
ssh cloud 'rm -rf runs_<old_run>'  # only when sure you're done
```

The replay buffer is also big (~hundreds of MB compressed). Worth pruning when a run is abandoned.

### Cost meter runs even when idle

Vast.ai etc. charge per hour the box is up, regardless of GPU utilization. If you finish work, **tear down the instance**. For YinshML's $0.40-1.04/hr range, an idle box overnight is $5-12 you didn't intend to spend.

## Recommended cadence for a real multi-day run

This session compressed everything into 24-hour cycles. For a multi-day run, the rhythm shifts:

- **Hour 0**: kick off training. Schedule wakeups every 30 min for the first 2 hours to catch early failures.
- **Hours 2-12**: drop to 60 min wakeups. Long iters can sometimes go 30+ min and don't need ultra-tight monitoring.
- **Hours 12+**: hourly is fine. Most of what's interesting happens at iter boundaries.
- **At day boundaries**: write a summary commit (`git commit + push`). Day N+1 starts cold-cache; commit notes are how Claude knows what happened on day N.
- **Always have an absolute stop time**: `if elapsed > 11h, stop scheduling new wakeups, write final summary`. Otherwise you'll burn a session on monitoring something that's plateaued.

Total cost of monitoring: each wakeup is one prompt-cache miss (~$0.05–0.15). At 30-min cadence over 24 hours = 48 wakeups × $0.10 ≈ $5. Cheap relative to compute.

## Antipatterns to avoid

1. **Deciding mid-experiment to extend / change parameters via re-running.** Decide before launch, log the decision, let it run. Saving a few hours by impatient kills usually costs more in confused state.
2. **Chasing too many hypotheses in parallel.** One config at a time per box. The whole point of the wakeup loop is patient seriality.
3. **Letting Claude commit untested code.** Always run regression tests after a behavior-changing patch lands, before launching another long experiment that depends on the patch.
4. **Forgetting to push the branch before launching.** Cloud must `git pull` to see new configs. Easy to forget; check `git log --oneline -3` on the cloud first.
5. **Trusting the watch log over the actual files.** The log is Claude's interpretation. The actual `metrics.json` / state_dict / log on disk is ground truth. When in doubt, grep the source files.

## Knowing when to stop

The autonomous loop is good for: probes that take 1-12 hours and have a binary "did the result come out as predicted" answer.

Stop spawning new experiments when:
- You're chasing the same hypothesis with diminishing returns (5+ probes on the same axis with no new signal)
- The remaining wall-time before user-return < (one probe's wall-time + 1 hour buffer for writeup)
- The compute budget is past 80% spent
- You've found the answer (positive or negative)

Write up. Stop. Let the user pick the next direction.

## Real numbers from this project

- Total wakeups across 2 days: ~80
- API cost of monitoring loop: ~$8-12
- Total cloud spend: ~$30 across both 24-hour windows
- Experiments launched: ~12 distinct training runs + ~15 evaluation runs
- Useful results: 1 root-cause bug fix (BN-stat-trash), 1 secondary bug fix (EMA rebind), 1 negative-result recipe knob proven robust (discrimination_weight=0), and a clean experimental matrix demonstrating that no yaml-level knob unlocks past-mimicry training

The methodology paid for itself many times over relative to what 30 hours of attended human monitoring would have cost.
