"""
SincNet / Gammatone genetic hyperparameter search.

Evolves GRU and FFN classifiers on top of raw-filter featurizers (SincNet and
Gammatone).  Same dataset, fitness function, and GA machinery as
train_micro_genetic.py — results are directly comparable.

Differences from the micro-genetic script:
  - featurizer_type locked to ["sincnet", "gammatone"]
  - arch limited to ["gru", "ffn"] (CNN not compatible with raw-filter outputs)
  - n_features = filter count (not MFCC bins) — meaningful values are higher
  - estimate_params uses correct GRU/FFN formula (no CNN path needed)
  - experiment name defaults to "sincnet_genetic" → separate MLflow experiment
    and output dir, but dataset is always "hey_mycroft"

Usage::

    .venv/bin/python train_sincnet_genetic.py
    .venv/bin/python train_sincnet_genetic.py --budget-minutes 240 --resume
"""
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Env / credentials ─────────────────────────────────────────────────────────
from ww_trainer.env import load_env, env_default
from ww_trainer.utils import read_dataset_csv
load_env()

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

import psutil
if psutil.virtual_memory().available / 1e9 < 8:
    sys.exit("ERROR: less than 8 GB RAM available — aborting")
if psutil.disk_usage("/").free / 1e9 < 20:
    sys.exit("ERROR: less than 20 GB disk free — aborting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sincnet_genetic")

import resource as _resource
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
_target = min(_hard, 65536)
if _soft < _target:
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
    logger.info("Raised RLIMIT_NOFILE: %d → %d", _soft, _target)

# ── Args ───────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--budget-minutes", type=float,
                    default=env_default("WW_BUDGET_MINUTES", 360,    float))
parser.add_argument("--stage1-pop",    type=int,
                    default=env_default("WW_STAGE1_POP",    12,     int))
parser.add_argument("--stage1-gen",    type=int,
                    default=env_default("WW_STAGE1_GEN",    5,      int))
parser.add_argument("--stage1-epochs", type=int,
                    default=env_default("WW_STAGE1_EPOCHS", 20,     int))
parser.add_argument("--stage2-pop",    type=int,
                    default=env_default("WW_STAGE2_POP",    8,      int))
parser.add_argument("--stage2-gen",    type=int,
                    default=env_default("WW_STAGE2_GEN",    4,      int))
parser.add_argument("--stage2-epochs", type=int,
                    default=env_default("WW_STAGE2_EPOCHS", 40,     int))
parser.add_argument("--final-epochs",  type=int,
                    default=env_default("WW_FINAL_EPOCHS",  100,    int))
parser.add_argument("--fp-weight",     type=float,
                    default=env_default("WW_FP_WEIGHT",     0.6,    float))
parser.add_argument("--param-budget",  type=int,
                    default=env_default("WW_PARAM_BUDGET",  60_000, int),
                    help="Soft param budget (SincNet/Gammatone GRUs are slightly larger)")
parser.add_argument("--top-k",         type=int,
                    default=env_default("WW_TOP_K",         3,      int))
parser.add_argument("--resume", action="store_true",
                    help="Skip completed stages; load existing all_results.json")
args = parser.parse_args()

# Dataset always comes from hey_mycroft for comparability.
# Output goes to a separate experiment dir so results don't mix.
DATASET_NAME    = env_default("WW_DATASET_NAME",    "hey_mycroft")
EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "sincnet_genetic")

BASE        = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR = Path(f"experiments/{DATASET_NAME}") / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
NWW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"
AUG_DIR     = DATASET_DIR / "augmentation"
OUT_BASE         = BASE / "sincnet_genetic"
PLOTS_DIR        = OUT_BASE / "plots"
ALL_RESULTS_JSON = OUT_BASE / "all_results.json"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ── Search space ───────────────────────────────────────────────────────────────
# featurizer_type is locked to raw-filter extractors.
# n_features = number of learned filters (not MFCC bins — higher counts useful).
# arch: GRU handles the temporal context well; FFN as a lighter baseline.
SEARCH_SPACE = {
    # ── Architecture ──────────────────────────────────────────────────
    "featurizer_type":  ["sincnet", "gammatone"],
    "arch":             ["gru", "ffn"],
    "n_features":       [40, 60, 80, 120],   # filter count
    "hidden_dim":       [32, 64, 128],
    "dropout":          [0.0, 0.1, 0.2, 0.3],

    # ── Optimisation ──────────────────────────────────────────────────
    "lr":               [5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    "batch_size":       [8, 16],

    # ── Imbalance weapons ─────────────────────────────────────────────
    "loss":             ["bce", "focal", "label_smoothing_bce", "rppl"],
    "mine_fraction":    [0.2, 0.33, 0.5, 0.7],
    "mixup_alpha":      [0.2, 0.5, 1.0, 2.0],
    "max_neg_weight":   [10.0, 30.0, 50.0, 100.0, 200.0],
    "spec_augment":     [True, False],
    "use_mixup":        [True, False],
}

# ── Dataset ────────────────────────────────────────────────────────────────────

def _build_dataset() -> tuple[list, list]:
    if not TRAIN_CSV.exists():
        sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_hey_mycroft.py first.")

    train_data = read_dataset_csv(TRAIN_CSV)
    test_data  = read_dataset_csv(TEST_CSV)

    if NWW_DIR.exists():
        nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
        rng = random.Random(42)
        rng.shuffle(nww_wavs)
        split = int(len(nww_wavs) * 0.8)
        train_data += [(str(p), "0") for p in nww_wavs[:split]]
        test_data  += [(str(p), "0") for p in nww_wavs[split:]]
        logger.info("Added %d NWW samples", len(nww_wavs))
    else:
        logger.warning("NWW subset not found at %s — training without it", NWW_DIR)

    pos = sum(1 for _, l in train_data if l == "1")
    neg = sum(1 for _, l in train_data if l == "0")
    ratio = neg / max(1, pos)
    logger.info("Dataset: %d train (%d pos / %d neg, ratio %.1f:1) | %d test",
                len(train_data), pos, neg, ratio, len(test_data))
    return train_data, test_data


# ── Fitness ────────────────────────────────────────────────────────────────────

def compute_fitness(f1: float, fpr: float, fnr: float, n_params: int) -> float:
    recall = 1.0 - fnr
    if recall < 0.70:
        return 0.0

    fp_score = 1.0 - args.fp_weight * fpr

    size_ratio = n_params / max(1, args.param_budget)
    if size_ratio > 2.0:
        return 0.0
    size_factor = 1.0 / (1.0 + max(0.0, size_ratio - 1.0))

    return max(0.0, recall * fp_score * size_factor)


# ── Param estimator ────────────────────────────────────────────────────────────

def estimate_params(config: dict) -> int:
    """Rough parameter count for SincNet/Gammatone + GRU/FFN."""
    n_feat = config.get("n_features", 80)   # filter outputs
    h      = config.get("hidden_dim", 64)
    arch   = config.get("arch", "gru")

    if arch == "gru":
        # GRU: 3 * h * (input + h) + output layer h * 1
        return 3 * h * (n_feat + h) + h
    else:  # ffn
        # 40 frames × n_feat flattened → h → h → 1
        return n_feat * 40 * h + h * h + h


# ── Evaluator ──────────────────────────────────────────────────────────────────
_FEAT_KWARGS = {
    "sincnet":   lambda n: {"featurizer_type": "sincnet",   "n_filters": n},
    "gammatone": lambda n: {"featurizer_type": "gammatone", "n_filters": n},
}


def evaluate_config(
    config: dict,
    train_data: list,
    test_data: list,
    trial_dir: Path,
    epochs: int,
    feature_cache=None,
    augment_opts: dict = None,
    export_onnx: bool = False,
) -> tuple[float, float, float, float, int]:
    """Train one config and return (fitness, f1, fpr, fnr, n_params)."""
    from ww_trainer.trainer import WakeWordTrainer
    from ww_trainer.evaluation import evaluate_model

    ft     = config.get("featurizer_type", "sincnet")
    n_feat = config.get("n_features", 80)
    feat_kwargs = _FEAT_KWARGS[ft](n_feat)

    loss_name  = config.get("loss", "bce")
    losses_cfg = [{"name": loss_name, "weight": 1.0}]
    trial_dir.mkdir(parents=True, exist_ok=True)

    try:
        trainer = WakeWordTrainer(
            arch=config.get("arch", "gru"),
            featurizer="",
            feature_dim=None,
            hidden_dim=config.get("hidden_dim", 64),
            dropout=config.get("dropout", 0.1),
            device="cpu",
            losses_cfg=losses_cfg,
            export_onnx=export_onnx,
            seed=42,
            wake_word=EXPERIMENT_NAME,
            mlflow_uri=MLFLOW_URI,
            **feat_kwargs,
            **(augment_opts or {}),
        )

        trainer.train(
            output_dir=trial_dir,
            train_data=train_data,
            test_data=test_data,
            epochs=epochs,
            batch_size=config.get("batch_size", 16),
            lr=config.get("lr", 5e-4),
            mine_fraction=config.get("mine_fraction", 0.33),
            max_neg_weight=config.get("max_neg_weight", 50.0),
            use_mixup=config.get("use_mixup", True),
            mixup_alpha=config.get("mixup_alpha", 1.0),
            spec_augment=config.get("spec_augment", True),
            neg_weight_schedule="linear",
            patience=max(5, epochs // 6),
            feature_cache=feature_cache,
            tsne_every=0, pca_every=0, umap_every=0,
        )

        result = evaluate_model(
            trainer.model, test_data, trainer.device,
            batch_size=128, threshold=0.5, feature_cache=feature_cache,
        )
        _, _, _, f1, _, _, _, _, targets, _, probs, report = result

        eer_thresh  = report.eer_threshold
        probs_arr   = np.array(probs)
        targets_arr = np.array(targets)
        preds_eer   = (probs_arr >= eer_thresh).astype(int)
        n_pos = int((targets_arr == 1).sum())
        n_neg = int((targets_arr == 0).sum())
        fp    = int(((preds_eer == 1) & (targets_arr == 0)).sum())
        fn    = int(((preds_eer == 0) & (targets_arr == 1)).sum())
        fpr   = fp / max(1, n_neg)
        fnr   = fn / max(1, n_pos)

        n_params = sum(p.numel() for p in trainer.model.parameters())
        fitness  = compute_fitness(f1, fpr, fnr, n_params)

        if export_onnx:
            try:
                trainer.model.load_checkpoint(str(trial_dir / "best_f1.pt"))
                trainer.model.feature_extractor.export_to_onnx(
                    str(trial_dir / "best_f1_featurizer.onnx")
                )
                logger.info("Exported featurizer ONNX to %s", trial_dir)
            except Exception as e:
                logger.warning("Featurizer ONNX export failed: %s", e)

        logger.info(
            "[trial] %s/%d arch=%s h=%d loss=%s mine=%.2f mixup=%s/%.1f "
            "spec=%s neg_w=%.0f → fitness=%.4f f1=%.4f fpr=%.3f fnr=%.3f params=%d",
            ft, n_feat, config.get("arch"), config.get("hidden_dim"), loss_name,
            config.get("mine_fraction", 0.33), config.get("use_mixup", True),
            config.get("mixup_alpha", 1.0), config.get("spec_augment", True),
            config.get("max_neg_weight", 50.0),
            fitness, f1, fpr, fnr, n_params,
        )
        return fitness, f1, fpr, fnr, n_params

    except Exception as exc:
        logger.warning("Trial failed: %s", exc, exc_info=True)
        return 0.0, 0.0, 1.0, 1.0, 0


# ── Genetic operators ──────────────────────────────────────────────────────────

def random_individual() -> dict:
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def crossover(p1: dict, p2: dict) -> dict:
    return {k: random.choice([p1[k], p2[k]]) for k in SEARCH_SPACE}


def mutate(ind: dict, mutation_rate: float) -> dict:
    m = ind.copy()
    for k in SEARCH_SPACE:
        if random.random() < mutation_rate:
            m[k] = random.choice(SEARCH_SPACE[k])
    return m


# ── One generation ─────────────────────────────────────────────────────────────

def run_generation(
    population: list[dict],
    train_data: list,
    test_data: list,
    epochs: int,
    out_dir: Path,
    trial_offset: int,
    all_results: list[dict],
    feature_cache=None,
    deadline: float = float("inf"),
    augment_opts: dict = None,
    stage: int = 0,
    mlf=None,
) -> list[dict]:
    evaluated_keys = {json.dumps(r["config"], sort_keys=True) for r in all_results}
    results = []
    for i, ind in enumerate(population):
        if time.monotonic() > deadline:
            logger.warning("Budget exhausted — skipping remaining %d trials", len(population) - i)
            break
        key = json.dumps(ind, sort_keys=True)
        if key in evaluated_keys:
            logger.info("Skipping duplicate config")
            continue
        tid = trial_offset + i
        fitness, f1, fpr, fnr, n_params = evaluate_config(
            ind, train_data, test_data,
            trial_dir=out_dir / f"trial_{tid}",
            epochs=epochs,
            feature_cache=feature_cache,
            augment_opts=augment_opts,
        )
        r = {"config": ind, "fitness": fitness, "f1": f1, "fpr": fpr, "fnr": fnr,
             "n_params": n_params, "tid": tid, "stage": stage}
        results.append(r)
        all_results.append(r)
        evaluated_keys.add(key)
        if mlf:
            mlf.log_trial(r)
        ALL_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        ALL_RESULTS_JSON.write_text(json.dumps(all_results, indent=2))
    return results


def select_next_population(
    results: list[dict],
    population_size: int,
    n_elite: int,
    mutation_rate: float,
) -> list[dict]:
    ranked  = sorted(results, key=lambda r: r["fitness"], reverse=True)
    elite   = [r["config"] for r in ranked[:n_elite]]
    next_pop = list(elite)
    pool = elite if len(elite) >= 2 else elite * 2
    while len(next_pop) < population_size:
        p1, p2 = random.sample(pool, 2)
        child = mutate(crossover(p1, p2), mutation_rate)
        next_pop.append(child)
    return next_pop


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_generation_curves(stage_histories: list[list[dict]], title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    for si, history in enumerate(stage_histories):
        label = f"Stage {si+1}"
        color = colors[si % len(colors)]
        gens = [h["generation"] for h in history]
        axes[0].plot(gens, [h["best_fitness"] for h in history], "-o", color=color,
                     label=f"{label} best", lw=2, ms=5)
        axes[0].plot(gens, [h["avg_fitness"]  for h in history], "--", color=color,
                     label=f"{label} avg",  lw=1, alpha=0.6)
        axes[1].plot(gens, [h["best_f1"]      for h in history], "-o", color=color,
                     label=label, lw=2, ms=5)
    axes[0].set_title("Fitness evolution"); axes[0].set_xlabel("Generation")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title("Best F1 per generation"); axes[1].set_xlabel("Generation")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.suptitle(title_prefix); fig.tight_layout()
    path = PLOTS_DIR / "generation_curves.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_all_trials(all_results: list[dict]) -> None:
    if not all_results:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fitnesses = [r["fitness"] for r in all_results]
    f1s  = [r["f1"]  for r in all_results]
    fprs = [r["fpr"] for r in all_results]
    tids = list(range(len(all_results)))
    axes[0].scatter(tids, fitnesses, c=fitnesses, cmap="RdYlGn", s=25, alpha=0.7)
    axes[0].set_title("Fitness across all trials"); axes[0].set_xlabel("Trial"); axes[0].grid(alpha=0.3)
    axes[1].scatter(fprs, f1s, c=fitnesses, cmap="RdYlGn", s=25, alpha=0.7)
    axes[1].set_title("F1 vs FPR"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("F1"); axes[1].grid(alpha=0.3)
    params = [estimate_params(r["config"]) for r in all_results]
    axes[2].scatter(params, fitnesses, c=fitnesses, cmap="RdYlGn", s=25, alpha=0.7)
    axes[2].axvline(args.param_budget, color="red", linestyle="--", alpha=0.5, label="Budget")
    axes[2].set_title("Fitness vs Param count"); axes[2].set_xlabel("Params (est)"); axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "all_trials.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("Saved: %s", path)


def plot_final_comparison(final_results: list[dict]) -> None:
    if not final_results:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, len(final_results) * 0.8 + 1)))
    sorted_r = sorted(final_results, key=lambda r: r["fitness"])
    labels = [
        f"{r['config'].get('featurizer_type')}/{r['config'].get('n_features')} "
        f"{r['config'].get('arch')}-h{r['config'].get('hidden_dim')} "
        f"{r['config'].get('loss')}"
        for r in sorted_r
    ]
    values = [r["fitness"] for r in sorted_r]
    colors = ["#2ecc71" if v == max(values) else "#3498db" for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, r in zip(bars, sorted_r):
        ax.text(r["fitness"] + 0.005, bar.get_y() + bar.get_height() / 2,
                f"F1={r['f1']:.3f} FPR={r['fpr']:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Fitness"); ax.set_title("Final models — SincNet/Gammatone genetic search")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout()
    path = PLOTS_DIR / "final_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    logger.info("Saved: %s", path)


from ww_trainer.ga_mlflow import GeneticMLflowLogger
from ww_trainer.visualization import plot_pareto_front, plot_config_analysis

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    random.seed(0)
    deadline   = time.monotonic() + args.budget_minutes * 60
    trial_counter = 0
    global_gen    = 0
    stage_histories: list[list[dict]] = []
    final_results:   list[dict] = []

    all_results: list[dict] = []
    if args.resume and ALL_RESULTS_JSON.exists():
        try:
            all_results = json.loads(ALL_RESULTS_JSON.read_text())
            trial_counter = max((r.get("tid", 0) for r in all_results), default=-1) + 1
            logger.info("Resumed: %d prior results (next tid=%d)", len(all_results), trial_counter)
        except Exception as exc:
            logger.warning("Could not load %s: %s — starting fresh", ALL_RESULTS_JSON, exc)

    train_data, test_data = _build_dataset()

    mlf = GeneticMLflowLogger(
        mlflow_uri=MLFLOW_URI,
        experiment_name=EXPERIMENT_NAME,
        run_name=f"sincnet_genetic · {args.budget_minutes:.0f}m",
        search_space=SEARCH_SPACE,
        search_params={
            "budget_minutes":  args.budget_minutes,
            "stage1_pop":      args.stage1_pop,
            "stage1_gen":      args.stage1_gen,
            "stage1_epochs":   args.stage1_epochs,
            "stage2_pop":      args.stage2_pop,
            "stage2_gen":      args.stage2_gen,
            "stage2_epochs":   args.stage2_epochs,
            "final_epochs":    args.final_epochs,
            "fp_weight":       args.fp_weight,
            "param_budget":    args.param_budget,
            "top_k":           args.top_k,
            "dataset":         DATASET_NAME,
        },
    )

    augment_opts = {}
    if any((AUG_DIR / "bg_noise").rglob("*.wav")):
        augment_opts["bg_noise_folder"] = str(AUG_DIR / "bg_noise")
    if any((AUG_DIR / "music").rglob("*.wav")):
        augment_opts["music_folder"] = str(AUG_DIR / "music")
    if any((AUG_DIR / "rir").rglob("*.wav")):
        augment_opts["rir_folder"] = str(AUG_DIR / "rir")
    if NWW_DIR.exists():
        augment_opts["wake_word_over_speech_folder"] = str(NWW_DIR)
    logger.info("Augmentation: %s", list(augment_opts.keys()) or "none")

    all_paths = list({p for p, _ in train_data + test_data})
    from ww_trainer.cache import SharedWaveformCache
    cache = SharedWaveformCache(sr=16000, max_duration=2.0)
    cache.preload(all_paths, n_workers=1, show_progress=True)
    logger.info("Cache: %s", cache.stats())

    s1_dir = OUT_BASE / "stage1"
    s2_dir = OUT_BASE / "stage2"

    # ── Stage 1: broad exploration ─────────────────────────────────────────────
    s1_best_json = s1_dir / "best_result.json"
    if args.resume and s1_best_json.exists():
        logger.info("Skipping Stage 1 (best_result.json exists)")
        s1_ranked = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
    else:
        logger.info("═══ STAGE 1: broad exploration — pop=%d gen=%d epochs=%d ═══",
                    args.stage1_pop, args.stage1_gen, args.stage1_epochs)
        population    = [random_individual() for _ in range(args.stage1_pop)]
        n_elite       = max(2, int(args.stage1_pop * 0.25))
        mutation_rate = 0.35
        s1_history:   list[dict] = []
        s1_results:   list[dict] = []

        for gen in range(args.stage1_gen):
            if time.monotonic() > deadline:
                logger.warning("Budget hit — stopping stage 1 early at gen %d", gen)
                break
            logger.info("── Stage 1 gen %d/%d ──", gen + 1, args.stage1_gen)
            results = run_generation(
                population, train_data, test_data,
                epochs=args.stage1_epochs,
                out_dir=s1_dir, trial_offset=trial_counter,
                all_results=all_results, feature_cache=cache,
                deadline=deadline, augment_opts=augment_opts,
                stage=1, mlf=mlf,
            )
            s1_results.extend(results)
            trial_counter += len(population)

            if not results:
                logger.warning("No results from gen %d — budget likely exhausted", gen)
                break
            best    = max(results, key=lambda r: r["fitness"])
            avg_fit = sum(r["fitness"] for r in results) / len(results)
            s1_history.append({"generation": gen, "best_fitness": best["fitness"],
                                "avg_fitness": avg_fit, "best_f1": max(r["f1"] for r in results)})
            logger.info("Gen %d best: fitness=%.4f f1=%.4f fpr=%.3f config=%s",
                        gen, best["fitness"], best["f1"], best["fpr"], best["config"])
            mlf.log_generation(1, gen, results, global_gen)
            global_gen += 1
            population    = select_next_population(results, args.stage1_pop, n_elite, mutation_rate)
            mutation_rate *= 0.85

        stage_histories.append(s1_history)
        s1_ranked = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
        s1_dir.mkdir(parents=True, exist_ok=True)
        s1_dir.joinpath("best_result.json").write_text(
            json.dumps({"best_config": s1_ranked[0]["config"],
                        "best_fitness": s1_ranked[0]["fitness"],
                        "best_f1": s1_ranked[0]["f1"]}, indent=2)
        )
        logger.info("Stage 1 complete. Best: %s", s1_ranked[0]["config"])

    # ── Stage 2: refinement ────────────────────────────────────────────────────
    s2_best_json = s2_dir / "best_result.json"
    if args.resume and s2_best_json.exists():
        logger.info("Skipping Stage 2 (best_result.json exists)")
        s2_ranked = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
    else:
        logger.info("═══ STAGE 2: refinement — pop=%d gen=%d epochs=%d ═══",
                    args.stage2_pop, args.stage2_gen, args.stage2_epochs)

        seen_keys: set[str] = set()
        seed_configs = []
        for r in s1_ranked:
            key = json.dumps(r["config"], sort_keys=True)
            if key not in seen_keys:
                seen_keys.add(key)
                seed_configs.append(r["config"])
            if len(seed_configs) >= 5:
                break

        population    = seed_configs + [mutate(random.choice(seed_configs), 0.2)
                                        for _ in range(args.stage2_pop - len(seed_configs))]
        n_elite       = max(2, int(args.stage2_pop * 0.30))
        mutation_rate = 0.15
        s2_history:   list[dict] = []
        s2_results:   list[dict] = []

        for gen in range(args.stage2_gen):
            if time.monotonic() > deadline:
                logger.warning("Budget hit — stopping stage 2 early at gen %d", gen)
                break
            logger.info("── Stage 2 gen %d/%d ──", gen + 1, args.stage2_gen)
            results = run_generation(
                population, train_data, test_data,
                epochs=args.stage2_epochs,
                out_dir=s2_dir, trial_offset=trial_counter,
                all_results=all_results, feature_cache=cache,
                deadline=deadline, augment_opts=augment_opts,
                stage=2, mlf=mlf,
            )
            s2_results.extend(results)
            trial_counter += len(population)

            if not results:
                logger.warning("No results from gen %d — budget likely exhausted", gen)
                break
            best    = max(results, key=lambda r: r["fitness"])
            avg_fit = sum(r["fitness"] for r in results) / len(results)
            s2_history.append({"generation": gen, "best_fitness": best["fitness"],
                                "avg_fitness": avg_fit, "best_f1": max(r["f1"] for r in results)})
            logger.info("Gen %d best: fitness=%.4f f1=%.4f fpr=%.3f config=%s",
                        gen, best["fitness"], best["f1"], best["fpr"], best["config"])
            mlf.log_generation(2, gen, results, global_gen)
            global_gen += 1
            population    = select_next_population(results, args.stage2_pop, n_elite, mutation_rate)
            mutation_rate *= 0.80

        stage_histories.append(s2_history)
        s2_ranked = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
        s2_dir.mkdir(parents=True, exist_ok=True)
        s2_dir.joinpath("best_result.json").write_text(
            json.dumps({"best_config": s2_ranked[0]["config"],
                        "best_fitness": s2_ranked[0]["fitness"],
                        "best_f1": s2_ranked[0]["f1"]}, indent=2)
        )
        logger.info("Stage 2 complete. Best: %s", s2_ranked[0]["config"])

    # ── Plots after search ─────────────────────────────────────────────────────
    if stage_histories:
        plot_generation_curves(stage_histories, "SincNet/Gammatone genetic search")
    plot_all_trials(all_results)
    plot_pareto_front(all_results, param_budget=args.param_budget,
                      title="Pareto front — SincNet/Gammatone genetic search",
                      out_path=PLOTS_DIR / "pareto_front.png")
    plot_config_analysis(all_results, search_space=SEARCH_SPACE,
                         top_k=max(10, len(all_results) // 4),
                         out_path=PLOTS_DIR / "config_analysis.png")

    # ── Stage 3: final training ────────────────────────────────────────────────
    best_final_json = OUT_BASE / "best_final.json"
    if args.resume and best_final_json.exists():
        logger.info("Skipping Stage 3 (best_final.json exists)")
    elif time.monotonic() > deadline:
        logger.warning("No budget left for final training — exiting after search")
    else:
        logger.info("═══ STAGE 3: final training — top-%d configs × %d epochs ═══",
                    args.top_k, args.final_epochs)
        seen_keys = set()
        top_configs = []
        for r in sorted(all_results, key=lambda r: r["fitness"], reverse=True):
            key = json.dumps(r["config"], sort_keys=True)
            if key not in seen_keys:
                seen_keys.add(key)
                top_configs.append(r)
            if len(top_configs) >= args.top_k:
                break

        final_results = []
        for rank, r in enumerate(top_configs):
            if time.monotonic() > deadline:
                logger.warning("Budget hit — stopping final training after rank %d", rank)
                break
            logger.info("── Final model rank=%d ──", rank + 1)
            final_dir = OUT_BASE / "final" / f"rank_{rank+1}"
            fitness, f1, fpr, fnr, n_params = evaluate_config(
                r["config"], train_data, test_data,
                trial_dir=final_dir,
                epochs=args.final_epochs,
                feature_cache=cache,
                augment_opts=augment_opts,
                export_onnx=True,
            )
            final_results.append({"config": r["config"], "fitness": fitness, "f1": f1,
                                   "fpr": fpr, "fnr": fnr, "n_params": n_params})
            logger.info("Final rank=%d → fitness=%.4f f1=%.4f fpr=%.3f", rank+1, fitness, f1, fpr)

        plot_final_comparison(final_results)

        if final_results:
            best_f = max(final_results, key=lambda r: r["fitness"])
            best_final_json.write_text(
                json.dumps({"best_config": best_f["config"],
                            "best_fitness": best_f["fitness"],
                            "best_f1": best_f["f1"],
                            "best_fpr": best_f["fpr"],
                            "best_fnr": best_f["fnr"],
                            "n_params": best_f["n_params"]}, indent=2)
            )

    mlf.finalize(all_results, final_results,
                 plots_dir=PLOTS_DIR, results_json=ALL_RESULTS_JSON)

    # ── Summary table ──────────────────────────────────────────────────────────
    top10 = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[:10]
    print("\n" + "═" * 100)
    print("  SincNet/Gammatone Genetic Search — Top 10 configs")
    print("═" * 100)
    print(f"  {'Rank':<5} {'Fitness':>8} {'F1':>7} {'FPR':>7} {'FNR':>7} {'Params':>8}  Config")
    print("  " + "-" * 95)
    for i, r in enumerate(top10):
        cfg = r["config"]
        print(f"  {i+1:<5} {r['fitness']:>8.4f} {r['f1']:>7.4f} {r['fpr']:>7.4f} {r['fnr']:>7.4f} "
              f"{r.get('n_params', 0):>8d}  "
              f"{cfg.get('featurizer_type')}/{cfg.get('n_features')} "
              f"arch={cfg.get('arch')} h={cfg.get('hidden_dim')} "
              f"loss={cfg.get('loss')} mine={cfg.get('mine_fraction')} "
              f"mixup={cfg.get('use_mixup')}/{cfg.get('mixup_alpha')} "
              f"spec={cfg.get('spec_augment')} neg_w={cfg.get('max_neg_weight')}")
    print("═" * 100)
    logger.info("Done. Results: %s  Plots: %s  Models: %s",
                ALL_RESULTS_JSON.resolve(), PLOTS_DIR.resolve(), OUT_BASE.resolve())


if __name__ == "__main__":
    main()
