# Tournament Progress Check Instructions

## Quick Status Check

Run this command to check current progress:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
python -c "
import json, time
from pathlib import Path

file = Path('yinsh_ml/docs/validation_results/heuristic_vs_random.json')
if file.exists():
    data = json.load(open(file))
    m = data.get('metrics', {})
    g = m.get('total_games', 0)
    ts = data.get('timestamp', time.time())
    elapsed = time.time() - ts
    rate = g / elapsed if elapsed > 0 else 0
    remaining = 1000 - g
    eta = remaining / rate if rate > 0 else 0
    
    print(f'📊 Progress: {g}/1000 games completed')
    print(f'⏱️  Time elapsed: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)')
    print(f'⚡ Rate: {rate*60:.2f} games/minute ({rate*3600:.1f} games/hour)')
    print(f'⏳ Estimated time remaining: {eta/3600:.2f} hours ({eta/60:.1f} minutes)')
    print(f'✅ Win rate: {m.get(\"win_rate\", 0)*100:.1f}%')
else:
    print('❌ Tournament file not found')
"
```

## Check if Tournament is Still Running

```bash
ps aux | grep "run_final_validation" | grep -v grep
```

## Resume Tournament (if needed)

If the tournament stopped, you can resume it:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
python scripts/run_final_validation.py --num-games 1000 --output-dir yinsh_ml/docs/validation_results --resume
```

