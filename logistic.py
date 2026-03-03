import os
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,precision_recall_curve

import load_data as ld 


# ----------------------------
# Feature building
# ----------------------------
def build_features_and_data(z_path: str, matches_dir: str):
    z = ld.load_z_data(z_path)

    predictions = ld.to_numpy_2d(z["predictions"]).astype(int) 
    positives = z["positives_per_query"]
    distances = ld.to_numpy_2d(z["distances"]).astype(float)

    Q = predictions.shape[0]
    matches = ld.load_matches_dir(matches_dir, expected_num_queries=Q) # load inliers for each query 

    # is retrieval top-1 wrong?
    correct = ld.correct_at_1(predictions, positives) # if top-1 retrieval is correct
    y = (~correct).astype(int)  # 1 = wrong, 0 = correct y[q] = 1 if top-1 retrieval is wrong → this is what logistic learns to predict.

    # feature 1: inliers(top1) For each query: take inliers between query and top-1 retrieved candidate.
    inliers_top1 = np.array([float(matches[q][0]["num_inliers"]) for q in range(Q)], dtype=float)

    # feature 2: distance margin = d2 - d1 
    # f top-1 is much better than top-2, then d2 - d1 is large ⇒ retrieval confident
    # If top-1 and top-2 are close, margin small ⇒ ambiguous ⇒ more likely wrong
    d1 = distances[:, 0]
    d2 = distances[:, 1]
    margin = (d2 - d1).astype(float)
 
    X = np.stack([inliers_top1, margin], axis=1) # column 0: inliers_top1, column 1: margin

    # retrieval list
    pred_retrieval = predictions

    # always-rerank list (reorder by inliers)
    pred_reranked = ld.reranked_preds_from_inliers(predictions, matches)

    return X, y, positives, inliers_top1, pred_retrieval, pred_reranked


# ----------------------------
# Logistic adaptive policy
# ----------------------------
def adaptive_preds_from_prob(pred_retrieval: np.ndarray, pred_reranked: np.ndarray, p_wrong: np.ndarray, p0: float) -> np.ndarray:
    """Use reranked list if p_wrong > p0 else keep retrieval list."""
    out = pred_retrieval.copy()
    use = (p_wrong > p0) 
    out[use] = pred_reranked[use]
    return out.astype(int)


def sweep_cutoffs(p_wrong: np.ndarray, positives: list, pred_retrieval: np.ndarray, pred_reranked: np.ndarray, cutoffs: np.ndarray):
    r1_list, r5_list, r10_list, r20_list = [], [], [], []
    frac_list = []

    for p0 in cutoffs:
        preds_adapt = adaptive_preds_from_prob(pred_retrieval, pred_reranked, p_wrong, p0)

        r1_list.append(ld.recall_at_n(preds_adapt, positives, 1))
        r5_list.append(ld.recall_at_n(preds_adapt, positives, 5))
        r10_list.append(ld.recall_at_n(preds_adapt, positives, 10))
        r20_list.append(ld.recall_at_n(preds_adapt, positives, 20))

        frac_list.append(float(np.mean(p_wrong > p0)))

    return (np.array(r1_list), np.array(r5_list), np.array(r10_list), np.array(r20_list), np.array(frac_list))


def choose_best_tradeoff(cutoffs: np.ndarray, r1: np.ndarray, frac: np.ndarray, eps: float = 0.001):
    """
    Choose the cheapest cutoff whose R@1 is within eps of the best R@1.
    """
    best_r1 = np.max(r1)
    ok = r1 >= (best_r1 - eps)
    if not np.any(ok):
        idx = int(np.argmax(r1))
        return cutoffs[idx], r1[idx], frac[idx]
    # pick minimal rerank fraction
    idx = int(np.argmin(frac[ok]))
    return cutoffs[ok][idx], r1[ok][idx], frac[ok][idx]
    # Among all “almost-best” R@1 solutions, choose the one that reranks the least.


# ----------------------------
# Plot
# ----------------------------

def save_fig(plots_dir, name):
    if plots_dir is None:
        plt.show()
    else:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, name), dpi=200, bbox_inches="tight")
        plt.close()


def plot_curves(cutoffs, r1, r5, r10, r20, frac, title_prefix, plots_dir=None):
   
    plt.figure()
    plt.plot(cutoffs, r1, label="R@1")
    plt.plot(cutoffs, r5, label="R@5")
    plt.plot(cutoffs, r10, label="R@10")
    plt.plot(cutoffs, r20, label="R@20")
    plt.xlabel("probability cutoff p0 (rerank if p_wrong > p0)")
    plt.ylabel("Recall@K")
    plt.title(f"{title_prefix} — Recall@K vs cutoff")
    plt.legend()
    save_fig(plots_dir, f"{title_prefix}_recall_vs_cutoff.png")

    plt.figure()
    plt.plot(cutoffs, 100 * frac)
    plt.xlabel("probability cutoff p0")
    plt.ylabel("% queries reranked")
    plt.title(f"{title_prefix} — Rerank fraction vs cutoff")
    save_fig(plots_dir, f"{title_prefix}_rerank_frac_vs_cutoff.png")

def plot_accuracy_cost(frac_reranked, r1, title, plots_dir=None):
    plt.figure()
    plt.plot(100 * frac_reranked, 100 * r1)
    plt.xlabel("% queries reranked")
    plt.ylabel("Adaptive R@1 (%)")
    plt.title(title)
    save_fig(plots_dir, f"{title}_accuracy_vs_cost.png")

def plot_time_tradeoff(frac_reranked, t_rerank, t_retrieval, title, plots_dir=None):
    T_always = t_retrieval + t_rerank
    T_adapt = t_retrieval + frac_reranked * t_rerank
    savings = 1.0 - (T_adapt / T_always)

    plt.figure()
    plt.plot(100 * frac_reranked, 100 * savings)
    plt.xlabel("% queries reranked")
    plt.ylabel("% time saved vs always-rerank")
    plt.title(title)
    save_fig(plots_dir, f"{title}_time_savings.png")

def plot_pr_curve(y_true, scores, title_prefix, plots_dir=None):
    """
    y_true: 1 if wrong, 0 if correct
    scores: p_wrong (higher => more uncertain)
    """
    ap = average_precision_score(y_true, scores)  
    precision, recall, _ = precision_recall_curve(y_true, scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} — PR curve (AUPRC={ap:.3f})")

    save_fig(plots_dir, f"{title_prefix}_pr_curve.png")
    return ap

def plot_uncertainty_hist(y_true, p_wrong, title_prefix, plots_dir=None, bins=30):
    y_true = np.asarray(y_true).astype(int)
    p_wrong = np.asarray(p_wrong).astype(float)

    p_correct = p_wrong[y_true == 0]
    p_wrong_vals = p_wrong[y_true == 1]

    plt.figure()
    plt.hist(
        p_correct,
        bins=bins,
        alpha=0.6,
        color="blue",
        edgecolor="black",
        label=f"Correct (y=0), n={len(p_correct)}"
    )
    plt.hist(
        p_wrong_vals,
        bins=bins,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label=f"Wrong (y=1), n={len(p_wrong_vals)}"
    )
    plt.xlabel("uncertainty score p_wrong")
    plt.ylabel("count")
    plt.title(f"{title_prefix} — p_wrong distribution (correct vs wrong)")
    plt.legend()

    save_fig(plots_dir, f"{title_prefix}_pwrong_hist.png")



# ----------------------------
# Main
# ----------------------------
def print_block(name, positives, pred_retrieval, pred_reranked, pred_adapt, frac, t_rerank, t_retrieval):
    r1_r = ld.recall_at_n(pred_retrieval, positives, 1)
    r5_r = ld.recall_at_n(pred_retrieval, positives, 5)
    r10_r = ld.recall_at_n(pred_retrieval, positives, 10)
    r20_r = ld.recall_at_n(pred_retrieval, positives, 20)

    r1_a = ld.recall_at_n(pred_reranked, positives, 1)
    r5_a = ld.recall_at_n(pred_reranked, positives, 5)
    r10_a = ld.recall_at_n(pred_reranked, positives, 10)
    r20_a = ld.recall_at_n(pred_reranked, positives, 20)

    r1_ad = ld.recall_at_n(pred_adapt, positives, 1)
    r5_ad = ld.recall_at_n(pred_adapt, positives, 5)
    r10_ad = ld.recall_at_n(pred_adapt, positives, 10)
    r20_ad = ld.recall_at_n(pred_adapt, positives, 20)

    print(f"\n[{name}]")
    print(f"Retrieval-only: R@1={100*r1_r:.2f}  R@5={100*r5_r:.2f}  R@10={100*r10_r:.2f}  R@20={100*r20_r:.2f}")
    print(f"Always-rerank:  R@1={100*r1_a:.2f}  R@5={100*r5_a:.2f}  R@10={100*r10_a:.2f}  R@20={100*r20_a:.2f}")
    print(f"Adaptive:       R@1={100*r1_ad:.2f}  R@5={100*r5_ad:.2f}  R@10={100*r10_ad:.2f}  R@20={100*r20_ad:.2f}")
    print(f"% reranked: {100*frac:.1f}")
    print(f"Est. time/query: {t_retrieval + frac*t_rerank:.3f}s (always-rerank: {t_retrieval + t_rerank:.3f}s)")


def main(args):
    # TRAIN
    X_tr1, y_tr1, _, _, _, _ = build_features_and_data(args.train_z_path, args.train_matches_dir)
    X_tr2, y_tr2, _, _, _, _ = build_features_and_data(args.train2_z_path, args.train2_matches_dir)
    X_tr = np.concatenate([X_tr1, X_tr2], axis=0)
    y_tr = np.concatenate([y_tr1, y_tr2], axis=0)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")) # Balanced weights help because wrong queries may be minority or majority depending on dataset
    ])
    model.fit(X_tr, y_tr)


    # VALIDATION
    X_va, y_va, pos_va, inl_va, pred_va, rer_va = build_features_and_data(args.val_z_path, args.val_matches_dir)
    p_wrong_va = model.predict_proba(X_va)[:, 1]
    cutoffs = np.linspace(0.0, 1.0, args.num_cutoffs)
    r1_va, r5_va, r10_va, r20_va, frac_va = sweep_cutoffs(p_wrong_va, pos_va, pred_va, rer_va, cutoffs)
    # choose p0* based on R@1 tradeoff
    p0_star, r1_star, frac_star = choose_best_tradeoff(cutoffs, r1_va, frac_va, eps=args.tradeoff_eps)
    # adaptive predictions at chosen cutoff (VAL)
    pred_adapt_va = adaptive_preds_from_prob(pred_va, rer_va, p_wrong_va, p0_star)
 
    if args.plot:
        plot_curves(cutoffs, r1_va, r5_va, r10_va, r20_va, frac_va,title_prefix=f"{args.name} (VAL)", plots_dir=args.plots_dir)
        plot_accuracy_cost(frac_va, r1_va, title=f"{args.name} (VAL) — Accuracy–cost trade-off", plots_dir=args.plots_dir)
        plot_time_tradeoff(frac_va, args.t_rerank, args.t_retrieval, title=f"{args.name} (VAL) time savings", plots_dir=args.plots_dir)
        plot_uncertainty_hist(y_va, p_wrong_va, f"{args.name}_VAL", plots_dir=args.plots_dir)



    print("\n=== VALIDATION (fixed p0* chosen on VAL) ===")
    auprc_va = plot_pr_curve(y_va, p_wrong_va, f"{args.name}_VAL", plots_dir=args.plots_dir)
    print(f"[{args.name}] VAL AUPRC - Uncertainty = {auprc_va:.3f}")
    print("Random baseline (positive rate) =", np.mean(y_va))

    print(f"Chosen cutoff p0*: {p0_star:.3f}  (eps={args.tradeoff_eps}, based on R@1)")
    print_block(f"{args.name} VAL", pos_va, pred_va, rer_va, pred_adapt_va, frac_star, args.t_rerank, args.t_retrieval)

    # TEST
    X_te, y_te, pos_te, inl_te, pred_te, rer_te = build_features_and_data(args.test_z_path, args.test_matches_dir)
    p_wrong_te = model.predict_proba(X_te)[:, 1]
    pred_adapt_te = adaptive_preds_from_prob(pred_te, rer_te, p_wrong_te, p0_star)
    frac_te = float(np.mean(p_wrong_te > p0_star))
    
    print("\n=== TEST (using p0* from VAL) ===")

    auprc_te = plot_pr_curve(y_te, p_wrong_te, f"{args.name}_TEST", plots_dir=args.plots_dir)
    print(f"[{args.name}] TEST AUPRC - Uncertainty = {auprc_te:.3f}")
    print("Random baseline (positive rate) =", np.mean(y_te))
    plot_uncertainty_hist(y_te, p_wrong_te, f"{args.name}_TEST", plots_dir=args.plots_dir)

    print_block(f"{args.name} TEST", pos_te, pred_te, rer_te, pred_adapt_te, frac_te, args.t_rerank, args.t_retrieval)



def build_arg_parser():
    p = argparse.ArgumentParser("Logistic adaptive reranking (inliers_top1 + distance margin) with Recall@K reporting")

    p.add_argument("--name", type=str, default="Run", help="Label for printing/plots")

    p.add_argument("--train_z_path", type=str, required=True)
    p.add_argument("--train_matches_dir", type=str, required=True)

    p.add_argument("--train2_z_path", type=str, required=True)
    p.add_argument("--train2_matches_dir", type=str, required=True)

    p.add_argument("--val_z_path", type=str, required=True)
    p.add_argument("--val_matches_dir", type=str, required=True)

    p.add_argument("--test_z_path", type=str, required=True)
    p.add_argument("--test_matches_dir", type=str, required=True)

    p.add_argument("--t_rerank", type=float, required=True, help="Matching/re-ranking cost per query (seconds)")
    p.add_argument("--t_retrieval", type=float, default=0.0, help="Retrieval-only cost per query (seconds)")

    p.add_argument("--num_cutoffs", type=int, default=101)
    p.add_argument("--tradeoff_eps", type=float, default=0.001)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--plots_dir", type=str, default=None)

  
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
