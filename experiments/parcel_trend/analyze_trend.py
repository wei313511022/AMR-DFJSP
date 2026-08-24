"""Parcel-count trend: extend_GNN+PPO (gc1, 3 seeds) vs dispatching rules at n = 20..100.

Fills in the gap between the 60 / 120 / 240 points of
`experiments/policy_vs_rules` stage `scaling`, which measured a sign flip somewhere
between n=60 (policy wins) and n=120 (policy loses) but had no resolution inside that
interval. 10 instances per size, so this is a trend probe, not a powered comparison --
every CI here is wide and is meant to be read as such.

Conventions are inherited wholesale from `analyze_policy.py` (pairing on
`(amrs, instance)`, seeds averaged WITHIN instance before across, never averaging over
`nu > 0` rows), which is imported rather than re-implemented so the two analyses cannot
drift apart.

    python analyze_trend.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE.parent / "policy_vs_rules"))

from analyze_policy import (  # noqa: E402
    load, pair, paired_stats, policy_table, rank_rules, rule_table, seed_averaged,
    variance_decomposition, write_csv,
)

SIZES = (20, 40, 60, 80, 100)
AMRS = 16
ARM = "gc1"
# Pre-registered baseline from experiments/policy_vs_rules/selection.json -- chosen on
# val_60 alone, and also the rule the policy was trained against. Carried over unchanged
# so this trend is not a fresh rule search dressed up as a comparison.
BASELINE = ("milk_run", "earliest_completion")


def policy_by_seed(n: int) -> dict:
    pol = policy_table(load(RAW / f"policy_n{n}.jsonl"), AMRS)
    by_seed = {k: v for k, v in pol.items() if f"_{ARM}_" in k and k.endswith("_best")}
    if len(by_seed) != 3:
        raise SystemExit(f"n={n}: expected 3 gc1 _best run keys, got {sorted(by_seed)}")
    return by_seed


def main() -> None:
    trend, per_seed_rows, rule_rank_rows = [], [], []

    for n in SIZES:
        rules = rule_table(load(RAW / f"rules_n{n}.jsonl"), AMRS)
        pol = policy_by_seed(n)

        # Strongest rule AT THIS n, recomputed per size: the whole question is whether the
        # rule that wins at n=60 keeps winning as the floor gets busier.
        #
        # `rank_rules` averages over cleanly-routed rows only, so a rule that fails to
        # route the hard instances would be ranked on the easy ones alone and look
        # spuriously strong. Only combinations that routed every instance are eligible to
        # be called "strongest"; the rest still appear in rule_ranking.csv with their
        # clean/instances counts.
        ranked = rank_rules(rules)
        for r in ranked:
            rule_rank_rows.append({"n_jobs": n, **r})
        eligible = [r for r in ranked if r["clean"] == r["instances"]]
        if not eligible:
            raise SystemExit(f"n={n}: no rule combination routed all instances")
        best_combo = tuple(eligible[0]["combo"].split("+", 1))

        row = {"n_jobs": n, "parcels_per_robot": round(n / AMRS, 2)}
        for tag, combo in (("baseline", BASELINE), ("best_rule", best_combo)):
            agg = seed_averaged(pol, rules[combo])
            row[f"{tag}_rule"] = "+".join(combo)
            for k in ("policy_mean", "rule_mean", "delta", "delta_ci95", "delta_pct",
                      "boot_lo", "boot_hi", "win_rate_pct", "perm_p", "n_pairs",
                      "significant"):
                row[f"{tag}_{k}"] = agg[k]

        var = variance_decomposition(pol)
        row["between_seed_sd"] = var["between_seed_sd"]
        row["policy_unroutable"] = sum(1 for p in pol.values()
                                       for r in p.values() if r["nu"] > 0)
        trend.append(row)

        for key, by_inst in sorted(pol.items()):
            kept, _ = pair(by_inst, rules[BASELINE], f"n={n} {key}")
            st = paired_stats(by_inst, rules[BASELINE], kept)
            per_seed_rows.append({"n_jobs": n, "run_key": key, **st})

    write_csv(HERE / "trend.csv", trend)
    write_csv(HERE / "trend_by_seed.csv", per_seed_rows)
    write_csv(HERE / "rule_ranking.csv", rule_rank_rows)

    out = []
    out.append("\nPARCEL-COUNT TREND -- m=16, greedy, no local improvement, 10 instances/size")
    out.append(f"policy = extend_GNN + PPO, arm {ARM}, seeds 42/43/44 (_best), "
               f"trained only at n=60")
    out.append("=" * 104)
    out.append(f"vs {'+'.join(BASELINE)}  [pre-registered baseline, = the policy's own "
               f"training baseline]")
    out.append(f"  {'n':>5} {'n/robot':>8} {'policy':>9} {'rule':>9} {'delta':>17} "
               f"{'delta%':>8} {'win%':>6} {'perm p':>8} {'seed sd':>8} {'nu>0':>5}")
    for r in trend:
        out.append(f"  {r['n_jobs']:>5} {r['parcels_per_robot']:>8.2f} "
                   f"{r['baseline_policy_mean']:>9.1f} {r['baseline_rule_mean']:>9.1f} "
                   f"{r['baseline_delta']:>+8.2f} +/-{r['baseline_delta_ci95']:<5.2f} "
                   f"{r['baseline_delta_pct']:>+7.2f}% {r['baseline_win_rate_pct']:>5.1f}% "
                   f"{r['baseline_perm_p']:>8.4f} {r['between_seed_sd']:>8.2f} "
                   f"{r['policy_unroutable']:>5}")

    out.append("")
    out.append("vs the strongest rule AT EACH n  [re-selected per size, post-hoc by "
               "construction]")
    out.append(f"  {'n':>5} {'best rule':>30} {'policy':>9} {'rule':>9} {'delta':>17} "
               f"{'delta%':>8} {'win%':>6} {'perm p':>8}")
    for r in trend:
        out.append(f"  {r['n_jobs']:>5} {r['best_rule_rule']:>30} "
                   f"{r['best_rule_policy_mean']:>9.1f} {r['best_rule_rule_mean']:>9.1f} "
                   f"{r['best_rule_delta']:>+8.2f} +/-{r['best_rule_delta_ci95']:<5.2f} "
                   f"{r['best_rule_delta_pct']:>+7.2f}% "
                   f"{r['best_rule_win_rate_pct']:>5.1f}% {r['best_rule_perm_p']:>8.4f}")

    out.append("")
    out.append("PER SEED, vs the pre-registered baseline")
    out.append(f"  {'n':>5} {'run_key':>20} {'policy':>9} {'delta':>17} {'win%':>6} "
               f"{'perm p':>8}")
    for r in per_seed_rows:
        out.append(f"  {r['n_jobs']:>5} {r['run_key']:>20} {r['policy_mean']:>9.1f} "
                   f"{r['delta']:>+8.2f} +/-{r['delta_ci95']:<5.2f} "
                   f"{r['win_rate_pct']:>5.1f}% {r['perm_p']:>8.4f}")

    out.append("")
    out.append("TOP 5 RULE COMBINATIONS PER SIZE (executed makespan)")
    for n in SIZES:
        top = [r for r in rule_rank_rows if r["n_jobs"] == n][:5]
        out.append(f"  n={n}: " + " | ".join(f"{r['combo']} {r['executed']:.1f}"
                                             for r in top))

    out.append("")
    out.append("10 instances per size: CIs are wide by design. Read the SIGN and the SLOPE")
    out.append("of delta%, not the individual p-values.")

    text = "\n".join(out)
    (HERE / "trend_summary.txt").write_text(text + "\n")
    print(text)
    plot(trend, HERE / "fig_parcel_trend.pdf")


def plot(trend: list, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ns = [r["n_jobs"] for r in trend]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax1.errorbar(ns, [r["baseline_policy_mean"] for r in trend], marker="o", ms=4, lw=1.4,
                 color="#185FA5", label="extend_GNN + PPO (3 seeds)")
    ax1.plot(ns, [r["baseline_rule_mean"] for r in trend], marker="s", ms=4, lw=1.4,
             color="#993C1D", label="+".join(BASELINE))
    ax1.plot(ns, [r["best_rule_rule_mean"] for r in trend], marker="^", ms=4, lw=1.2,
             ls="--", color="#4A4A4A", label="strongest rule at each $n$")
    ax1.set_xlabel("parcels per instance")
    ax1.set_ylabel("executed makespan")
    ax1.set_title("Absolute makespan", fontsize=9)
    ax1.legend(fontsize=6.5, frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    d = [r["baseline_delta_pct"] for r in trend]
    e = [100 * r["baseline_delta_ci95"] / r["baseline_rule_mean"] for r in trend]
    ax2.axhline(0, color="#993C1D", lw=1.0)
    ax2.errorbar(ns, d, yerr=e, marker="o", ms=4, lw=1.4, capsize=2.5, color="#185FA5")
    ax2.fill_between(ns, 0, d, where=[x < 0 for x in d], color="#185FA5", alpha=0.12)
    ax2.set_xlabel("parcels per instance")
    ax2.set_ylabel("policy $-$ rule (% of rule)")
    ax2.set_title("Trained at $n=60$; advantage vs workload", fontsize=9)
    ax2.grid(alpha=0.25, lw=0.5)

    for ax in (ax1, ax2):
        ax.set_xticks(ns)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
