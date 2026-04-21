"""Run the value-head SAE probe end-to-end (Track B §8).

Pipeline:
  1. Load a trained checkpoint as a NetworkWrapper.
  2. Generate `--num-positions` positions via in-process self-play
     (no MCTS — we want raw network behavior at each position).
  3. Capture penultimate value-head activations for every position.
  4. Train a Sparse Autoencoder on those activations (256 → 2048 features).
  5. Per-feature analysis: top-20 positions per feature, dead-feature counts,
     confident-error position list.
  6. Persist everything under `--output-dir` for hand-labeling.

Recommended checkpoint per the prompt audit:
  runs/supervised_warmstart_v2/iteration_6/checkpoint_iteration_6.pt
  (the +61 ELO peak from the warm-start run, stronger than the bake-off
  challenger which was a statistical tie with baseline at the toy scale.)

Usage:
  python scripts/run_sae_probe.py \
    --checkpoint runs/supervised_warmstart_v2/iteration_6/checkpoint_iteration_6.pt \
    --num-positions 10000 \
    --output-dir analysis_output/sae_probe/iter6 \
    --device mps

For a smoke run (fast, no real signal) drop --num-positions to 200 and
--epochs to 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Imports moved to lazy fns to keep --help fast.
logger = logging.getLogger('sae_probe')


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to a NetworkWrapper-loadable checkpoint.')
    p.add_argument('--output-dir', type=str, required=True,
                   help='Directory for activations, SAE, feature report.')
    p.add_argument('--num-positions', type=int, default=10000,
                   help='Number of positions to capture (default: 10000).')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device: cpu | mps | cuda. Default cpu (network forward only).')
    p.add_argument('--use-enhanced-encoding', action='store_true',
                   help='Match the checkpoint`s training encoder.')
    p.add_argument('--temperature', type=float, default=0.5,
                   help='Move-sampling temperature for self-play.')
    p.add_argument('--epochs', type=int, default=50,
                   help='SAE training epochs.')
    p.add_argument('--l1-coefficient', type=float, default=1e-3,
                   help='SAE L1 sparsity coefficient.')
    p.add_argument('--learning-rate', type=float, default=1e-3,
                   help='SAE Adam learning rate.')
    p.add_argument('--batch-size', type=int, default=256,
                   help='SAE training batch size.')
    p.add_argument('--top-k', type=int, default=20,
                   help='Per-feature top-K positions for the report.')
    p.add_argument('--confidence-threshold', type=float, default=0.7,
                   help='Threshold for "confidently wrong" prediction analysis. '
                        'A network with discrimination near 0 (RESEARCH_LOG.md) '
                        'never exceeds 0.7 — try 0.05 or 0.1 for low-confidence '
                        'networks to surface the tail-of-distribution mistakes.')
    p.add_argument('--seed', type=int, default=20260419,
                   help='RNG seed (position generation + SAE init).')
    p.add_argument('--verbose', action='store_true',
                   help='Enable DEBUG logging.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    # Lazy imports so --help is fast (full ML stack is heavy).
    from yinsh_ml.interpretability.activation_capture import ValueHeadActivationCapture
    from yinsh_ml.interpretability.feature_analysis import FeatureAnalyzer, save_report
    from yinsh_ml.interpretability.position_generator import generate_positions
    from yinsh_ml.interpretability.sae import SAEConfig, SparseAutoencoder, save_sae, train_sae
    from yinsh_ml.network.wrapper import NetworkWrapper

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # ---- Step 1+2: position generation (self-play with the checkpoint) ----
    logger.info(f"Generating {args.num_positions} positions from {args.checkpoint}")
    t0 = time.time()
    encoded, network_values, outcomes, move_numbers = generate_positions(
        checkpoint_path=args.checkpoint,
        num_positions=args.num_positions,
        device=args.device,
        use_enhanced_encoding=args.use_enhanced_encoding,
        temperature=args.temperature,
        seed=args.seed,
    )
    logger.info(
        f"Generated {len(encoded)} positions in {time.time()-t0:.1f}s "
        f"(value range [{network_values.min():.3f}, {network_values.max():.3f}], "
        f"outcome dist: win={float((outcomes>0).mean()):.2f}, "
        f"loss={float((outcomes<0).mean()):.2f}, draw={float((outcomes==0).mean()):.2f})"
    )

    # Persist position-level metadata (small; useful for the analyst).
    np.save(output_dir / 'positions_states.npy', encoded)
    np.save(output_dir / 'positions_network_values.npy', network_values)
    np.save(output_dir / 'positions_outcomes.npy', outcomes)
    np.save(output_dir / 'positions_move_numbers.npy', move_numbers)

    # ---- Step 3: activation capture ----
    logger.info("Capturing value-head penultimate activations")
    network = NetworkWrapper(
        device=torch.device(args.device),
        use_enhanced_encoding=args.use_enhanced_encoding,
    )
    network.load_model(args.checkpoint)
    capture = ValueHeadActivationCapture(network, device=args.device)

    # Process in chunks to keep peak memory bounded.
    chunk = 256
    for start in range(0, len(encoded), chunk):
        capture.capture(encoded[start:start + chunk])
    activations, _ = capture.stack()
    np.save(output_dir / 'activations.npy', activations)
    logger.info(f"Captured activations shape: {activations.shape}")

    # ---- Step 4: SAE training ----
    cfg = SAEConfig(
        input_dim=activations.shape[1],
        expansion=8,
        l1_coefficient=args.l1_coefficient,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    logger.info(
        f"Training SAE: input_dim={cfg.input_dim}, num_features={cfg.num_features}, "
        f"epochs={cfg.epochs}, l1_coef={cfg.l1_coefficient}"
    )
    sae = SparseAutoencoder(cfg)
    stats = train_sae(sae, activations, device=args.device, verbose=True)
    save_sae(sae, stats, output_dir)
    logger.info(
        f"Final SAE: recon_loss={stats.recon_loss[-1]:.5f}, "
        f"sparsity={stats.sparsity[-1]:.3f}, "
        f"dead_features={stats.dead_feature_count[-1]}/{cfg.num_features}"
    )

    # ---- Step 5: feature analysis ----
    logger.info("Computing per-feature top-K positions and confident errors")
    analyzer = FeatureAnalyzer(sae, device=args.device)
    report = analyzer.per_feature_summary(activations, top_k=args.top_k)
    save_report(report, output_dir / 'feature_report.json')
    logger.info(
        f"Feature report: {report.num_features} features, "
        f"{report.dead_feature_count} dead, "
        f"{report.sparse_feature_count} sparse (<1%), "
        f"{report.dense_feature_count} dense (>50%)"
    )

    confident_errors = FeatureAnalyzer.find_confident_errors(
        network_values, outcomes, confidence_threshold=args.confidence_threshold
    )
    np.save(output_dir / 'confident_error_indices.npy', confident_errors)
    logger.info(
        f"Confident-error positions: {len(confident_errors)}/{len(network_values)} "
        f"({100*len(confident_errors)/max(1,len(network_values)):.1f}%) — "
        f"these are the inputs to the 'what concept did the value head miss?' analysis"
    )

    # ---- Final summary file ----
    summary = {
        'checkpoint': str(args.checkpoint),
        'num_positions': int(len(encoded)),
        'sae_config': cfg.__dict__,
        'final_recon_loss': float(stats.recon_loss[-1]),
        'final_sparsity': float(stats.sparsity[-1]),
        'final_dead_features': int(stats.dead_feature_count[-1]),
        'feature_report_summary': {
            'dead': report.dead_feature_count,
            'sparse_lt_1pct': report.sparse_feature_count,
            'dense_gt_50pct': report.dense_feature_count,
        },
        'confident_errors': {
            'count': int(len(confident_errors)),
            'fraction_of_positions': float(len(confident_errors) / max(1, len(network_values))),
        },
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary to {output_dir/'summary.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
