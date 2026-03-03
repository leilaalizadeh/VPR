import os
import re
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import load_data as ld


# ----------------------------
# Adaptive reranking
# ----------------------------

def adaptive_rerank_preds(pred_retrieval: np.ndarray, pred_reranked: np.ndarray, inliers_top1: np.ndarray, threshold: float) -> np.ndarray:
    """
      if inliers_top1 < threshold -> use reranked list
      else -> use retrieval list
    """
    use_rerank = inliers_top1 < threshold   # True → query is “hard” → rerank
                                            # False → query is “easy” → keep retrieval
    out = pred_retrieval.copy()             # Start with the cheap output (retrieval)
    out[use_rerank] = pred_reranked[use_rerank] # For the subset of hard queries, replace the whole top-K list with the reranked one
    return out.astype(int)
# hard: if inliers < t then rerank

# def adaptive_rerank_top1(predictions_2d: np.ndarray, reranked_top1: np.ndarray, inliers_top1: np.ndarray, threshold: float) -> np.ndarray:
#     top1_retrieval = predictions_2d[:, 0].astype(int) # Extracts the retrieval top-1 id for each query.
#     use_rerank = inliers_top1 < threshold
#     out = np.where(use_rerank, reranked_top1, top1_retrieval)
#     return out.astype(int)


def sweep_thresholds(pred_retrieval: np.ndarray, pred_reranked: np.ndarray, positives_per_query: list, inliers_top1: np.ndarray, thresholds: np.ndarray):
    r1_list, r5_list, r10_list, r20_list = [], [], [], []
    frac_list = []

    for t in thresholds:
        preds_adapt = adaptive_rerank_preds(pred_retrieval, pred_reranked, inliers_top1, t)
        # Compute Recall@N from the adaptive predictions
        r1_list.append(ld.recall_at_n(preds_adapt, positives_per_query, 1))
        r5_list.append(ld.recall_at_n(preds_adapt, positives_per_query, 5))
        r10_list.append(ld.recall_at_n(preds_adapt, positives_per_query, 10))
        r20_list.append(ld.recall_at_n(preds_adapt, positives_per_query, 20))

        frac_list.append(float(np.mean(inliers_top1 < t)))  # # Compute Recall@N from the adaptive predictions

    return (np.array(r1_list), np.array(r5_list), np.array(r10_list), np.array(r20_list),
            np.array(frac_list))



# ----------------------------
# 4) Plots
# ----------------------------
def save_fig(plots_dir, name):
    if plots_dir is None:
        plt.show()
    else:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, name), dpi=200, bbox_inches="tight")
        plt.close()

def plot_inliers_hist(plots_dir: str,inliers_top1: np.ndarray, correct: np.ndarray, title: str):
    correct_inl = inliers_top1[correct]
    wrong_inl = inliers_top1[~correct]

    plt.figure()
    plt.hist(correct_inl, bins=30, alpha=0.7, label=f"Correct @1 (n={len(correct_inl)})")
    plt.hist(wrong_inl, bins=30, alpha=0.7, label=f"Wrong @1 (n={len(wrong_inl)})")
    plt.xlabel("num_inliers (top-1 retrieved pair)")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    #plt.show()
    save_fig(plots_dir, f"{title}_hist.png")


def plot_sweep(plots_dir: str, thresholds: np.ndarray, r1: np.ndarray, frac_reranked: np.ndarray, title_prefix: str):
    plt.figure()
    print(thresholds)
    print(r1)
    plt.plot(thresholds, r1)
    plt.xlabel("inliers threshold t")
    plt.ylabel("Adaptive R@1")
    plt.title(f"{title_prefix} — R@1 vs threshold")
    #plt.show()
    save_fig(plots_dir,f"{title_prefix}_r1_vs_threshold.png")

    plt.figure()
    plt.plot(thresholds, 100 * frac_reranked)
    plt.xlabel("inliers threshold t")
    plt.ylabel("% queries reranked")
    plt.title(f"{title_prefix} — Rerank fraction vs threshold")
    #plt.show()
    save_fig(plots_dir,f"{title_prefix}_rerank_frac_vs_threshold.png")

def plot_time_tradeoff(plots_dir:str, frac_reranked: np.ndarray, t_rerank: float, t_retrieval: float = 0.0, title: str = "Time per query tradeoff"):
    
    T_always = t_retrieval + t_rerank
    T_adapt = t_retrieval + frac_reranked * t_rerank
    savings = 1.0 - (T_adapt / T_always)

    plt.figure()
    plt.plot(100 * frac_reranked, 100 * savings)
    plt.xlabel("% queries reranked")
    plt.ylabel("% time saved vs always-rerank")
    plt.title(title)
    #plt.show()
    
    save_fig(plots_dir,f"{title}_time_savings.png")

def plot_accuracy_cost(frac_reranked, r1, title, plots_dir=None, prefix="run"):

    plt.figure()
    plt.plot(100*frac_reranked, 100*r1)
    plt.xlabel("% queries reranked")
    plt.ylabel("Adaptive R@1 (%)")
    plt.title(title)

    save_fig(plots_dir,f"{prefix}_accuracy_vs_cost.png")


# ----------------------------
# 5) Main runner
# ----------------------------

def run_adaptive(z_path: str, matches_dir: str, dataset_name: str, t_rerank: float, t_retrieval: float,threshold_mode: str,num_thresholds: int, fixed_threshold: float, plots_dir: str):
    z = ld.load_z_data(z_path)
    predictions = ld.to_numpy_2d(z["predictions"])
    positives = z["positives_per_query"]

    num_q = predictions.shape[0]
    matches = ld.load_matches_dir(matches_dir, expected_num_queries=num_q) #one per query 

    if len(matches) != predictions.shape[0]:
        raise ValueError(
            f"Num match files ({len(matches)}) != num queries in predictions ({predictions.shape[0]}). "
            f"Ensure matches_dir has exactly one file per query in correct order."
        )

    correct = ld.correct_at_1(predictions, positives)   # retrieval top-1 correct or not
    inliers_top1 = ld.get_inliers_top1(matches) 

    pred_retrieval = predictions.astype(int)                 
    pred_reranked  = ld.reranked_preds_from_inliers(pred_retrieval, matches)  

   
    r1_retrieval  = ld.recall_at_n(pred_retrieval, positives, 1)
    r5_retrieval  = ld.recall_at_n(pred_retrieval, positives, 5)
    r10_retrieval = ld.recall_at_n(pred_retrieval, positives, 10)
    r20_retrieval = ld.recall_at_n(pred_retrieval, positives, 20)

    r1_reranked  = ld.recall_at_n(pred_reranked, positives, 1)
    r5_reranked  = ld.recall_at_n(pred_reranked, positives, 5)
    r10_reranked = ld.recall_at_n(pred_reranked, positives, 10)
    r20_reranked = ld.recall_at_n(pred_reranked, positives, 20)

    if fixed_threshold is not None: #test datasets
        preds_adapt = adaptive_rerank_preds(pred_retrieval, pred_reranked, inliers_top1, fixed_threshold)

        r1_adapt  = ld.recall_at_n(preds_adapt, positives, 1)
        r5_adapt  = ld.recall_at_n(preds_adapt, positives, 5)
        r10_adapt = ld.recall_at_n(preds_adapt, positives, 10)
        r20_adapt = ld.recall_at_n(preds_adapt, positives, 20)

        frac = float(np.mean(inliers_top1 < fixed_threshold))

        print("\nFixed threshold evaluation:")
        print(f"  t = {fixed_threshold:.2f}")
        print(f"  Adaptive: R@1={100*r1_adapt:.2f} R@5={100*r5_adapt:.2f} R@10={100*r10_adapt:.2f} R@20={100*r20_adapt:.2f}")
        print(f"  % reranked    = {100*frac:.1f}")
        print(f"  Estimated time/query = {t_retrieval + frac*t_rerank:.3f}s (always-rerank: {t_retrieval + t_rerank:.3f}s)")
        return


    print(f"\n[{dataset_name}]")
    print(f"Retrieval-only: R@1={100*r1_retrieval:.2f}  R@5={100*r5_retrieval:.2f}  R@10={100*r10_retrieval:.2f}  R@20={100*r20_retrieval:.2f}")
    print(f"Always-rerank:  R@1={100*r1_reranked:.2f}  R@5={100*r5_reranked:.2f}  R@10={100*r10_reranked:.2f}  R@20={100*r20_reranked:.2f}")
    print(f"Inliers(top1): min={inliers_top1.min():.1f}, median={np.median(inliers_top1):.1f}, max={inliers_top1.max():.1f}")

    plot_inliers_hist(plots_dir,inliers_top1, correct, title=f"{dataset_name}: inliers(top1) correct vs wrong")

    
    if threshold_mode == "quantiles":
        qs = np.linspace(0.0, 1.0, num_thresholds)
        thresholds = np.unique(np.quantile(inliers_top1, qs))
    elif threshold_mode == "range":
        lo = float(np.floor(inliers_top1.min()))
        hi = float(np.ceil(inliers_top1.max()))
        thresholds = np.linspace(lo, hi, num_thresholds)
    else:
        raise ValueError("threshold_mode must be one of: quantiles, range")

    
    r1_adapt, r5_adapt, r10_adapt, r20_adapt, frac_reranked = sweep_thresholds(pred_retrieval, pred_reranked, positives, inliers_top1, thresholds)
    
    plot_sweep(plots_dir,thresholds, r1_adapt, frac_reranked, title_prefix=dataset_name)
    plot_accuracy_cost(frac_reranked, r1_adapt, f"{dataset_name} — Accuracy–cost trade-off", plots_dir=plots_dir, prefix=dataset_name)

    best_r1 = np.max(r1_adapt) # Find the maximum achievable R@1.
    cands = np.where(np.isclose(r1_adapt, best_r1, atol=1e-12))[0] # Find all thresholds that achieve this same best R@1 (ties)
    best_idx = int(cands[np.argmin(frac_reranked[cands])]) # Among them, pick the one with minimum rerank fraction (cheapest).

   
    best_t = float(thresholds[best_idx])
    best_r1 = float(r1_adapt[best_idx])
    best_frac = float(frac_reranked[best_idx])
    
   
    print("\nBest threshold (max R@1):")
    print(f"  t = {best_t:.2f}")
    print(f"  Adaptive: R@1={100*r1_adapt[best_idx]:.2f} R@5={100*r5_adapt[best_idx]:.2f} @10={100*r10_adapt[best_idx]:.2f} R@20={100*r20_adapt[best_idx]:.2f}")
    print(f"  % reranked    = {100*best_frac:.1f}")
    print(f"  Estimated time/query = {t_retrieval + best_frac*t_rerank:.3f}s (always-rerank: {t_retrieval + t_rerank:.3f}s)")



def build_arg_parser():
    p = argparse.ArgumentParser(description="Adaptive reranking using num_inliers on top-1.")
    p.add_argument("--z_path", type=str, required=True, help="Path to z_data.torch")
    p.add_argument("--matches_dir", type=str, required=True, help="Folder containing per-query match .torch files")
    p.add_argument("--dataset_name", type=str, default="Dataset")

    p.add_argument("--t_rerank", type=float, required=True, help="Reranking time per query for the matching model.")
    p.add_argument("--t_retrieval", type=float, default=0.0, help="Optional retrieval-only time per query.")

    p.add_argument("--threshold_mode", type=str, choices=["quantiles", "range"], default="quantiles", help="How to create thresholds: quantiles or range.")
    p.add_argument("--num_thresholds", type=int, default=41, help="Number of thresholds to sweep (default 41).")
    p.add_argument("--fixed_threshold", type=float, default=None, help="If set, skip sweep and evaluate adaptive rerank using this threshold.")
    p.add_argument("--plots_dir", type=str, default=None, help="If set, save plots as PNGs into this folder.")
    
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    run_adaptive(
        z_path=args.z_path,
        matches_dir=args.matches_dir,
        dataset_name=args.dataset_name,
        t_rerank=args.t_rerank,
        t_retrieval=args.t_retrieval,
        threshold_mode=args.threshold_mode,
        num_thresholds=args.num_thresholds,
        fixed_threshold=args.fixed_threshold,
        plots_dir = args.plots_dir
    )
