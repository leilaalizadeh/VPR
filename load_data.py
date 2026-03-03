import os
import re #regular expression
import argparse #lets you run script from terminal z_path
import torch
import numpy as np
import math

import matplotlib.pyplot as plt


def load_z_data(z_path: str) -> dict: #prediction vpr_model
    z = torch.load(z_path, map_location="cpu", weights_only=False)
    if not isinstance(z, dict): #check format
        raise TypeError(f"Expected dict in {z_path}, got {type(z)}")
    for k in ["predictions", "positives_per_query"]: #keys 'database_utms', 'positives_per_query', 'predictions', 'distances'
        if k not in z:
            raise KeyError(f"Missing key '{k}' in z_data. Found keys: {list(z.keys())}")
    return z

def natural_key(filename: str):
    m = re.search(r"(\d+)", filename) #12, filename 12.torch
    return (int(m.group(1)) if m else math.inf, filename)

def load_matches_dir(matches_dir: str, expected_num_queries: int, ext: str = ".torch") -> list: #image_matching model 
   
    files = [f for f in os.listdir(matches_dir) if f.endswith(ext)] #Lists all files in matches_dir that end with .torch.
    if not files:
        raise FileNotFoundError(f"No *{ext} files found in: {matches_dir}")

    idx_to_file = {}
    for f in files: #For each filename, extract a number (query id) like 0, 1, 2, … 
        m = re.search(r"(\d+)", f)
        if not m:
            continue
        idx = int(m.group(1))
        idx_to_file[idx] = f   #Store in dictionary: idx_to_file[query_id] = filename

    missing = [i for i in range(expected_num_queries) if i not in idx_to_file] #expected query indices are 0..expected_num_queries-1 
    if missing:
        raise ValueError(f"Missing match files for query indices: {missing[:20]} (showing up to 20)")

   
    all_matches = []
    for i in range(expected_num_queries):
        path = os.path.join(matches_dir, idx_to_file[i])
        obj = torch.load(path, map_location="cpu", weights_only=False) #create full path to file 

        if not isinstance(obj, list) or len(obj) == 0 or "num_inliers" not in obj[0]: #a list, list contains dicts, each dict has key "num_inliers"
            raise ValueError(f"{path} has unexpected format (need list of dicts with 'num_inliers')")

        all_matches.append(obj) #match resuts to query(i)

    extras = sorted([k for k in idx_to_file.keys() if k >= expected_num_queries]) #
    if extras:
        print(f"Ignoring extra match files with indices >= {expected_num_queries}: {extras[:10]}{'...' if len(extras)>10 else ''}", flush=True)

    return all_matches #list of K dicts (one per retrieved candidate).

def to_numpy_2d(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.array(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def correct_at_1(predictions_2d: np.ndarray, positives_per_query: list) -> np.ndarray:
    n = predictions_2d.shape[0] #number of queries 
    if len(positives_per_query) != n:
        raise ValueError(f"positives_per_query length {len(positives_per_query)} != num_queries {n}")

    correct = np.zeros(n, dtype=bool) #boolean array
    for q in range(n):
        top1 = int(predictions_2d[q, 0]) # Get top1 retrieved database index for query q.
        pos = positives_per_query[q] # Get ground truth positives for that query.Convert to python list if tensor
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        correct[q] = top1 in set(pos) # True if top1 id appears among positives.
    #retrieval top-1 correct or not
    return correct

# Extract the inlier count for the top-1 retrieved candidate (candidate rank 0).
def get_inliers_top1(matches_per_query: list) -> np.ndarray:
    n = len(matches_per_query)
    inliers = np.zeros(n, dtype=float)
    for q in range(n):
        inliers[q] = float(matches_per_query[q][0]["num_inliers"]) 
        #is the list of match dicts for query q.
        #[0] top_1 retireval
        # #inliers how many geo matching have found  
    return inliers

# Compute the best top-1 after reranking by choosing the candidate with maximum inliers.
def reranked_top1_from_inliers(predictions_2d: np.ndarray, matches_per_query: list) -> np.ndarray:
    n, topK = predictions_2d.shape
    reranked = np.zeros(n, dtype=int)

    for q in range(n):
        mlist = matches_per_query[q]
        k = min(topK, len(mlist)) #Some match files might have fewer than K candidates.
        inl = np.array([float(mlist[i]["num_inliers"]) for i in range(k)], dtype=float) # Create array of inliers for each candidate in top-K.
        best_i = int(np.argmax(inl)) # Index of maximum inliers.
        reranked[q] = int(predictions_2d[q, best_i]) # Choose the database id at that position in the retrieval list.
    return reranked # array length Q of reranked top1 database indices.

# Compute R@1 given a top-1 list.
def recall_at_1_from_top1(top1_db_idx: np.ndarray, positives_per_query: list) -> float:
    n = len(top1_db_idx)
    hits = 0
    for q in range(n):
        pos = positives_per_query[q]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        if int(top1_db_idx[q]) in set(pos):
            hits += 1
    return hits / n

# Compute Recall@N for full prediction lists.
def recall_at_n(predictions_2d: np.ndarray, positives_per_query: list, N: int) -> float:
    """Recall@N: % queries having at least one positive in top-N predictions."""
    hits = 0
    Q = predictions_2d.shape[0]
    for q in range(Q):
        topN = predictions_2d[q, :N].astype(int).tolist() # Take top-N predicted database indices.
        pos = positives_per_query[q]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        pos = set(pos)  # Check if any of topN is positive:
        if any(p in pos for p in topN):
            hits += 1
    return hits / Q
# This creates the full reranked top-K list (not only top1):
# For each query:
# compute inliers for each candidate
# sort candidates by inliers descending
# reorder predictions accordingly
def reranked_preds_from_inliers(predictions_2d: np.ndarray, matches_per_query: list) -> np.ndarray:
    """
    Always-rerank list: reorder the retrieved top-K candidates by inliers descending.
    Returns QxK reordered predictions.
    """
    Q, K = predictions_2d.shape
    out = np.zeros_like(predictions_2d, dtype=int)

    for q in range(Q):
        mlist = matches_per_query[q]
        k_eff = min(K, len(mlist))
        inl = np.array([float(mlist[i]["num_inliers"]) for i in range(k_eff)], dtype=float)
        order = np.argsort(-inl)  # descending inliers
        out[q, :k_eff] = predictions_2d[q, order].astype(int)
        if k_eff < K: # If match list shorter than K, copy rest unchanged:
            out[q, k_eff:] = predictions_2d[q, k_eff:].astype(int)

    return out