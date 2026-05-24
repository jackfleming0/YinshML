# Branch D.1 v2 — SUCCESS PATH
Generated at 2026-05-24T17:40:01Z

## Training
Run dir: runs_branchD1_v2/20260524_110301
Duration: 5.713556063888889 h
Iterations completed: 5 / 5
Promotions: 5
Final anchor WR: 1.0

## Anchor track
```
2026-05-24 14:07:04,734 - TrainingSupervisor - INFO - ANCHOR[mcts/64]: iter 2, 40/40 = 100.0%
ANCHOR[mcts/64]: iter 2, 40/40 = 100.0%
2026-05-24 14:53:30,822 - TrainingSupervisor - INFO - ANCHOR[raw]: iter 3, 38/40 = 95.0%
ANCHOR[raw]: iter 3, 38/40 = 95.0%
2026-05-24 15:26:32,667 - TrainingSupervisor - INFO - ANCHOR[mcts/64]: iter 3, 37/40 = 92.5%
ANCHOR[mcts/64]: iter 3, 37/40 = 92.5%
2026-05-24 16:13:28,123 - TrainingSupervisor - INFO - ANCHOR[raw]: iter 4, 40/40 = 100.0%
ANCHOR[raw]: iter 4, 40/40 = 100.0%
2026-05-24 16:45:44,641 - TrainingSupervisor - INFO - ANCHOR[mcts/64]: iter 4, 40/40 = 100.0%
ANCHOR[mcts/64]: iter 4, 40/40 = 100.0%
```

## Loss progression (from feedback.md)
```

## Iteration 1
- policy_loss: 2.990575733008208
- value_loss: 1.923278899104507
- value_accuracy: n/a
- avg_game_length: 70.57
- suggestions:self_play: {'num_simulations': 200, 'late_simulations': 100, 'final_temp': 0.1, 'games_per_iteration': 110}
- suggestions:trainer: {'value_head_lr_factor': 5.5}

## Iteration 2
- policy_loss: 2.7086970783629507
- value_loss: 1.9074443544981614
- value_accuracy: n/a
- avg_game_length: 69.01
- suggestions:self_play: {'num_simulations': 200, 'late_simulations': 100, 'final_temp': 0.1, 'games_per_iteration': 100}
- suggestions:trainer: {'value_head_lr_factor': 5.5}

## Iteration 3
- policy_loss: 2.4882495659816115
- value_loss: 1.8790147108367727
- value_accuracy: n/a
- avg_game_length: 68.25
- suggestions:self_play: {'num_simulations': 200, 'late_simulations': 100, 'final_temp': 0.1, 'games_per_iteration': 110}
- suggestions:trainer: {'value_head_lr_factor': 5.0}

## Iteration 4
- policy_loss: 2.3316021552452675
- value_loss: 1.8609084379978669
- value_accuracy: n/a
- avg_game_length: 66.15
- suggestions:self_play: {'num_simulations': 200, 'late_simulations': 100, 'final_temp': 0.1, 'games_per_iteration': 110}
- suggestions:trainer: {'value_head_lr_factor': 5.0}

## Iteration 5
- policy_loss: 2.172954789797465
- value_loss: 1.8574614408688668
- value_accuracy: n/a
- avg_game_length: 66.32
- suggestions:self_play: {'num_simulations': 200, 'late_simulations': 100, 'final_temp': 0.1, 'games_per_iteration': 110}
- suggestions:trainer: {'value_head_lr_factor': 5.0}
```

## SPRT verdict
```json
{
  "config": {
    "anchor": "models/branchC_volume_pretrain/best_iter_4.pt",
    "anchor_label": "best_iter_4",
    "candidates": [
      "runs_branchD1_v2/20260524_110301/iteration_4/checkpoint_iteration_4_ema.pt"
    ],
    "mode": "sprt",
    "num_games": 40,
    "num_simulations": 64,
    "opening_sample_plies": 20,
    "opening_temperature": 1.0,
    "sprt": {
      "p0": 0.5,
      "p1": 0.6,
      "alpha": 0.05,
      "beta": 0.05,
      "max_games": 400
    },
    "seed": 42,
    "device": "cuda",
    "engine": "self_play.MCTS.search_batch"
  },
  "results": [
    {
      "candidate_path": "runs_branchD1_v2/20260524_110301/iteration_4/checkpoint_iteration_4_ema.pt",
      "candidate_label": "iteration_4/checkpoint_iteration_4_ema",
      "anchor_path": "models/branchC_volume_pretrain/best_iter_4.pt",
      "candidate_wr": 0.0625,
      "candidate_wins": 1,
      "anchor_wins": 15,
      "draws": 0,
      "cand_white_wins": 0,
      "cand_black_wins": 1,
      "ci95_lo": 0.01111905730833121,
      "ci95_hi": 0.2832926836802987,
      "verdict": "NOT_STRONGER",
      "sprt": {
        "label": "iteration_4/checkpoint_iteration_4_ema",
        "decision": "NOT_STRONGER",
        "games": 16,
        "cand_wins": 1,
        "anchor_wins": 15,
        "draws": 0,
        "cand_white_wins": 0,
        "cand_black_wins": 1,
        "llr": -3.164831712919191,
        "llr_lower": -2.9444389791664403,
        "llr_upper": 2.9444389791664403,
        "wr_decisive": 0.0625,
        "ci95_lo": 0.01111905730833121,
        "ci95_hi": 0.2832926836802987,
        "p0": 0.5,
        "p1": 0.6,
        "alpha": 0.05,
        "beta": 0.05,
        "max_games": 400
      },
      "seconds": 454.53477239608765
    }
  ],
  "elapsed_seconds": 454.93609166145325
}```

## NEXT STEPS
Box is intentionally LEFT RUNNING. Pull files via:
```
scp -P 30981 root@79.116.87.141:/root/SUMMARY.md .
scp -P 30981 root@79.116.87.141:/root/YinshML/logs/branchD1_v2_iter4_vs_frozen.json logs/
scp -P 30981 root@79.116.87.141:/root/YinshML/runs_branchD1_v2/20260524_110301/iteration_4/checkpoint_iteration_4_ema.pt models/
```

Then terminate the instance manually via vast.ai dashboard.
