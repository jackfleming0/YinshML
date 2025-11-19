# Step-by-Step Instructions: Returning Tournament Results

## When Tournament Completes (or You Want to Stop Early)

Follow these steps to return the results to the AI assistant for final synthesis:

### Step 1: Verify Tournament Completion

Check if the tournament has finished:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
python -c "
import json
from pathlib import Path

file = Path('yinsh_ml/docs/validation_results/heuristic_vs_random.json')
if file.exists():
    data = json.load(open(file))
    m = data.get('metrics', {})
    g = m.get('total_games', 0)
    print(f'Games completed: {g}/1000')
    if g >= 1000:
        print('✅ Tournament complete!')
    else:
        print(f'⚠️  Tournament incomplete ({g}/1000 games)')
        print('You can still proceed with current results or wait for completion.')
else:
    print('❌ Tournament file not found')
"
```

### Step 2: Generate Final Report

Generate the comprehensive validation report:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
python scripts/generate_validation_report.py \
    --results-dir yinsh_ml/docs/validation_results \
    --output yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md
```

**Note:** The script automatically looks for `heuristic_vs_random.json` and `heuristic_vs_baseline.json` in the results directory.

This will create: `yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md`

### Step 3: Verify Output Files Exist

Confirm these files exist:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
ls -lh yinsh_ml/docs/validation_results/*.json yinsh_ml/docs/validation_results/*.md
```

You should see:
- `heuristic_vs_random.json` - Tournament results vs random
- `heuristic_vs_baseline.json` - Tournament results vs baseline
- `FINAL_VALIDATION_REPORT.md` - Generated report

### Step 4: Return Results to AI Assistant

**Option A: Share the report file directly**
- Open: `yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md`
- Copy its contents and share with the AI

**Option B: Share file paths**
- Tell the AI: "The tournament is complete. Please review the results in `yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md`"

**Option C: Quick summary**
- Run this to get a quick summary:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
python -c "
import json
from pathlib import Path

random_file = Path('yinsh_ml/docs/validation_results/heuristic_vs_random.json')
if random_file.exists():
    data = json.load(open(random_file))
    m = data.get('metrics', {})
    print('=== TOURNAMENT RESULTS SUMMARY ===')
    print(f'Total games: {m.get(\"total_games\", 0)}')
    print(f'Win rate: {m.get(\"win_rate\", 0)*100:.1f}%')
    print(f'Average move time: {m.get(\"average_move_time\", 0)*1000:.3f} ms')
    print(f'Max move time: {m.get(\"max_move_time\", 0)*1000:.3f} ms')
    print(f'Average game length: {m.get(\"average_game_length\", 0):.1f} moves')
"
```

### Step 5: What the AI Will Do Next

Once you share the results, the AI will:
1. ✅ Review the final validation report
2. ✅ Verify all success criteria are met
3. ✅ Mark Task 15 subtasks as complete
4. ✅ Mark Task 15 as done
5. ✅ Identify the next task to work on

## Quick Reference Commands

**Check progress:**
```bash
cd /Users/jackfleming/PycharmProjects/YinshML && python -c "import json, time; data = json.load(open('yinsh_ml/docs/validation_results/heuristic_vs_random.json')); m = data.get('metrics', {}); print(f'{m.get(\"total_games\", 0)}/1000 games')"
```

**Generate report:**
```bash
cd /Users/jackfleming/PycharmProjects/YinshML && python scripts/generate_validation_report.py --results-dir yinsh_ml/docs/validation_results --output yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md
```

**View report:**
```bash
cat yinsh_ml/docs/validation_results/FINAL_VALIDATION_REPORT.md
```

