import os, numpy as np, torch


# Average importance from XGBoost (GPU if available) + RandomForest (CPU)
def feature_selection_topk(X: np.ndarray, y: np.ndarray, keep_frac: float, subsample: int | None, rank_cfg: dict):
    n = len(X); idx = np.arange(n)
    if subsample is not None and subsample < n:
        rng = np.random.RandomState(42)
        idx = rng.choice(idx, size=subsample, replace=False)
    Xs, ys = X[idx], y[idx]


    imp_list = []


    # --- XGBoost ---
    if rank_cfg.get('xgboost', {}).get('enabled', True):
        try:
            import xgboost as xgb
            use_gpu = torch.cuda.is_available()
            params = rank_cfg['xgboost'].copy()
            params.setdefault('random_state', 42)
            params['tree_method'] = 'gpu_hist' if use_gpu else params.get('tree_method', 'hist')
            params['predictor'] = 'gpu_predictor' if use_gpu else params.get('predictor', 'auto')
            # remove flags that XGBClassifier may not accept as None
            params = {k:v for k,v in params.items() if v is not None}
            model = xgb.XGBClassifier(**params)
            model.fit(Xs, ys)
            imp_list.append(model.feature_importances_.astype(np.float64))
        except Exception as ex:
            print(f"[WARN] XGBoost ranking failed ({ex}). Skipping XGB.")


    # --- RandomForest (sklearn, CPU) ---
    if rank_cfg.get('random_forest', {}).get('enabled', True):
        from sklearn.ensemble import RandomForestClassifier
        rf_params = rank_cfg['random_forest'].copy()
        rf_params.setdefault('random_state', 42)
        model = RandomForestClassifier(**rf_params)
        model.fit(Xs, ys)
        imp_list.append(model.feature_importances_.astype(np.float64))


    if not imp_list:
        print("[WARN] No rankers available. Using uniform importances.")
        imp = np.ones(X.shape[1], dtype=np.float64)
    else:
        imp = np.mean(np.stack(imp_list, axis=0), axis=0)


    k = max(1, int(round(X.shape[1] * keep_frac)))
    topk_idx = np.argsort(imp)[::-1][:k]
    return np.sort(topk_idx)