# ============================================================
# Focused rolling single-ticker experiment block
# Purpose:
# - Start with AAPL only.
# - Use 3-year rolling train window + 1-year test window, repeated for latest 3 test years.
# - Run perfect-hindsight labels, RL/action labels, and current PnL labels.
# - Export per-date row-level predictions in simulator-friendly format.
# - Produce per-fold and overall metrics: R² where meaningful, within-ticker Spearman,
#   top 10% vs bottom 10% diagnostics, and classification metrics where applicable.
#
# Paste this block after the notebook has defined:
# frames, all_data, TARGET_CONFIGS, FEATURE_SET_CONFIGS,
# prepare_xy, filter_rows_for_target, get_models_for_task,
# regression_metrics, binary_metrics, multiclass_metrics, safe_spearman, safe_pearson,
# build_prediction_frame, register_table, OUTPUT_DIR, PREDICTION_DIR, TABLE_DIR.
# ============================================================

from sklearn.base import clone

ROLLING_SINGLE_TICKER_CONFIG = {
    # Start with AAPL. Later, use: ['AAPL', 'PG', 'NVDA', 'BAC']
    'tickers': ['AAPL'],
    'feature_set_name': 'combined_project_b',
    'target_names': [
        # Perfect hindsight labels
        'pi_hindsight_entry_long',
        'pi_hindsight_entry_positive',
        'pi_hindsight_entry_original',
        'pi_hindsight_entry_6bins',
        # RL/action labels
        'rl_expert_action',
        'rl_long_action_binary',
        'rl_long_is_best',
        'rl_long_action_quality',
        # Current PnL target
        'rl_long_current_pnl',
    ],
    # Keep this small first. Add ElasticNet/Ridge/Logistic later if needed.
    'regression_model_names': ['RandomForest', 'LightGBM'],
    'binary_model_names': ['Logistic', 'RandomForest', 'LightGBM'],
    'multiclass_model_names': ['LogisticMulticlass', 'RandomForest', 'LightGBM'],
    'train_window_years': 3,
    'test_window_years': 1,
    'n_test_folds': 3,
    'min_train_rows': 250,
    'min_test_rows': 40,
    'top_bottom_pct': 0.10,
    # Signal threshold is only for export direction. Metrics still use all rows.
    'long_signal_threshold': 0.90,
    # Set explicitly if you want a fixed period, e.g. [2023, 2024, 2025].
    # If None, the code uses the latest 3 feasible test years in the data.
    'test_years': None,
}


def _ensure_rolling_base_data(frames, all_data):
    """Ensure combined data has datetime and year columns."""
    if all_data is None or len(all_data) == 0:
        all_data = pd.concat(frames.values(), ignore_index=True, sort=False).copy()
    df = all_data.copy()
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
    if YEAR_COL not in df.columns or df[YEAR_COL].isna().all():
        if DATE_COL not in df.columns:
            raise ValueError(f'Missing both {YEAR_COL} and {DATE_COL}; cannot build yearly rolling folds.')
        df[YEAR_COL] = df[DATE_COL].dt.year
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce').astype('Int64')
    return df


def build_single_ticker_rolling_folds(df, ticker, train_window_years=3, n_test_folds=3, test_years=None):
    """
    Build 3-year rolling training windows followed by one-year test windows.
    Example: train 2020-2022 -> test 2023; train 2021-2023 -> test 2024.
    """
    one = df[df[TICKER_COL].astype(str) == str(ticker)].copy()
    years = sorted(one[YEAR_COL].dropna().astype(int).unique().tolist())
    if test_years is None:
        feasible = []
        for y in years:
            train_years = list(range(y - train_window_years, y))
            if all(ty in years for ty in train_years):
                feasible.append(y)
        test_years = feasible[-n_test_folds:]
    else:
        test_years = [int(y) for y in test_years]

    folds = []
    for test_year in test_years:
        train_years = list(range(test_year - train_window_years, test_year))
        folds.append({
            'ticker': ticker,
            'walkforward_scheme': f'rolling_{train_window_years}y',
            'window_label': f'{min(train_years)}-{max(train_years)}_to_{test_year}',
            'fold_id': f'{ticker}_rolling_{min(train_years)}_{max(train_years)}_test_{test_year}',
            'train_years': train_years,
            'train_start_year': min(train_years),
            'train_end_year': max(train_years),
            'n_train_years': len(train_years),
            'test_year': test_year,
        })
    return folds


def get_focused_model_names_for_task(task, config):
    if task == 'regression':
        return config['regression_model_names']
    if task == 'binary':
        return config['binary_model_names']
    if task == 'multiclass':
        return config['multiclass_model_names']
    return []


def _model_classes(model):
    """Return fitted classifier classes, including classes inside sklearn Pipeline."""
    if hasattr(model, 'classes_'):
        return list(model.classes_)
    if hasattr(model, 'named_steps'):
        last_step = list(model.named_steps.values())[-1]
        if hasattr(last_step, 'classes_'):
            return list(last_step.classes_)
    return None


def _positive_or_long_score(model, X, task, target_name=None):
    """
    Return a continuous score suitable for ranking.
    - Regression: caller should use y_pred.
    - Binary: probability of positive class when available.
    - Multiclass: probability of 'long' class for action targets; otherwise expected ordinal class if numeric.
    """
    if not hasattr(model, 'predict_proba'):
        if hasattr(model, 'decision_function'):
            z = model.decision_function(X)
            z = np.asarray(z)
            if z.ndim == 1:
                return 1 / (1 + np.exp(-z))
            return np.nanmax(z, axis=1)
        return None

    proba = model.predict_proba(X)
    classes = _model_classes(model)
    if classes is None:
        return proba[:, -1] if proba.ndim == 2 else proba

    classes_str = [str(c).lower() for c in classes]

    # For RL expert-action classification, rank by long-action probability when possible.
    if 'long' in classes_str:
        return proba[:, classes_str.index('long')]
    if '1' in classes_str and task in ['binary', 'multiclass']:
        return proba[:, classes_str.index('1')]
    if 1 in classes:
        return proba[:, classes.index(1)]
    if True in classes:
        return proba[:, classes.index(True)]

    # If classes are numeric ordinal labels, use expected class value as the ranking score.
    class_numeric = pd.to_numeric(pd.Series(classes), errors='coerce')
    if class_numeric.notna().all():
        return np.dot(proba, class_numeric.to_numpy(dtype=float))

    # Last fallback: confidence in predicted class.
    return np.max(proba, axis=1)


def make_ranking_target_values(y, target_name):
    """
    Convert actual y_true into a numeric value for ranking diagnostics.
    For action target, use long-vs-not-long indicator. For other labels, use numeric value.
    """
    s = pd.Series(y).copy()
    if target_name == 'rl_expert_action':
        return s.astype(str).str.lower().eq('long').astype(float)
    out = pd.to_numeric(s, errors='coerce')
    return out.astype(float)


def percentile_against_train_distribution(test_scores, train_scores):
    """
    Leakage-safe signal score: percentile of each test prediction relative to training-window predictions.
    This avoids ranking the test period using the test distribution itself.
    """
    train_scores = pd.to_numeric(pd.Series(train_scores), errors='coerce').dropna().sort_values().to_numpy()
    test_scores = pd.to_numeric(pd.Series(test_scores), errors='coerce').to_numpy()
    if len(train_scores) == 0:
        return pd.Series(np.nan, index=range(len(test_scores)))
    pct = np.searchsorted(train_scores, test_scores, side='right') / len(train_scores)
    return pd.Series(np.clip(pct, 0, 1))


def top_bottom_diagnostics(y_true, score, pct=0.10, target_name=None):
    """Compare realised outcomes in top-score and bottom-score groups."""
    y_rank = make_ranking_target_values(y_true, target_name)
    score = pd.to_numeric(pd.Series(score), errors='coerce')
    df = pd.DataFrame({'y_rank': y_rank, 'score': score}).dropna()
    if len(df) < 10 or df['score'].nunique() < 2:
        return {
            'top_n': 0, 'bottom_n': 0,
            'top_mean_true_rank_value': np.nan,
            'bottom_mean_true_rank_value': np.nan,
            'top_minus_bottom_mean_true': np.nan,
            'top_positive_rate': np.nan,
            'bottom_positive_rate': np.nan,
            'top_bottom_lift': np.nan,
        }
    n = max(1, int(np.ceil(len(df) * pct)))
    ranked = df.sort_values('score', ascending=False)
    top = ranked.head(n)
    bottom = ranked.tail(n)
    top_pos = (top['y_rank'] > 0).mean()
    bottom_pos = (bottom['y_rank'] > 0).mean()
    return {
        'top_n': int(len(top)),
        'bottom_n': int(len(bottom)),
        'top_mean_score': top['score'].mean(),
        'bottom_mean_score': bottom['score'].mean(),
        'top_mean_true_rank_value': top['y_rank'].mean(),
        'bottom_mean_true_rank_value': bottom['y_rank'].mean(),
        'top_minus_bottom_mean_true': top['y_rank'].mean() - bottom['y_rank'].mean(),
        'top_positive_rate': top_pos,
        'bottom_positive_rate': bottom_pos,
        'top_bottom_lift': top_pos / bottom_pos if bottom_pos and bottom_pos > 0 else np.nan,
    }


def focused_metrics_for_prediction_frame(pred, target_cfg, target_name, pct=0.10):
    """Calculate metrics for one fold or one aggregate group."""
    task = target_cfg['task']
    out = {
        'n': len(pred),
        'task': task,
        'target_name': target_name,
    }
    y_true = pred[TRUE_COL]
    y_pred = pred[PRED_COL]
    score = pred[SCORE_COL]
    y_rank = make_ranking_target_values(y_true, target_name)

    if task == 'regression':
        out.update(regression_metrics(y_true, y_pred))
    elif task == 'binary':
        out.update(binary_metrics(y_true, y_pred, y_score=score))
        # R² on probability/score is not a classification metric, but is useful as an auxiliary calibration-style diagnostic.
        yt_num = pd.to_numeric(pd.Series(y_true), errors='coerce')
        sc_num = pd.to_numeric(pd.Series(score), errors='coerce')
        valid = yt_num.notna() & sc_num.notna()
        out['r2_on_score_auxiliary'] = r2_score(yt_num[valid], sc_num[valid]) if valid.sum() >= 3 and yt_num[valid].nunique() > 1 else np.nan
    elif task == 'multiclass':
        out.update(multiclass_metrics(y_true, y_pred, y_score=None))

    out['within_ticker_spearman'] = safe_spearman(y_rank, score)
    out['within_ticker_pearson'] = safe_pearson(y_rank, score)
    out.update(top_bottom_diagnostics(y_true, score, pct=pct, target_name=target_name))
    return out


def fit_predict_focused_single_ticker_fold(all_data, fold, target_name, target_cfg, feature_cols, model_name, model, config):
    """Fit one ticker-year rolling fold and return metrics + row-level predictions."""
    ticker = fold['ticker']
    target_col = target_cfg['column']
    task = target_cfg['task']

    ticker_df = all_data[all_data[TICKER_COL].astype(str) == str(ticker)].copy()
    ticker_df = filter_rows_for_target(ticker_df, target_name)

    train_df = ticker_df[ticker_df[YEAR_COL].astype(int).isin(fold['train_years'])].copy()
    test_df = ticker_df[ticker_df[YEAR_COL].astype(int) == int(fold['test_year'])].copy()

    X_train, y_train = prepare_xy(train_df, feature_cols, target_col, task)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col, task)

    if len(y_train) < config['min_train_rows']:
        return None, None, {'status': 'skipped', 'reason': 'too_few_train_rows', 'n_train': len(y_train), **fold}
    if len(y_test) < config['min_test_rows']:
        return None, None, {'status': 'skipped', 'reason': 'too_few_test_rows', 'n_test': len(y_test), **fold}
    if task in ['binary', 'multiclass'] and pd.Series(y_train).nunique(dropna=True) < 2:
        return None, None, {'status': 'skipped', 'reason': 'single_class_train', 'n_train': len(y_train), **fold}

    fitted = clone(model)
    fitted.fit(X_train, y_train)

    y_pred = fitted.predict(X_test)
    train_pred = fitted.predict(X_train)

    if task == 'regression':
        score_test = y_pred
        score_train = train_pred
    else:
        score_test = _positive_or_long_score(fitted, X_test, task, target_name=target_name)
        score_train = _positive_or_long_score(fitted, X_train, task, target_name=target_name)
        if score_test is None:
            score_test = y_pred
        if score_train is None:
            score_train = train_pred

    pred = build_prediction_frame(
        test_df, y_test, y_pred, score_test,
        target_name=target_name,
        task=task,
        feature_set_name=config['feature_set_name'],
        model_name=model_name,
        split_name='rolling_test',
    )

    # Add fold metadata.
    for k, v in fold.items():
        if k != 'train_years':
            pred[k] = v
    pred['train_years'] = ','.join(map(str, fold['train_years']))

    # Leakage-safe signal score using only training-window score distribution.
    pred[SIGNAL_SCORE_COL] = percentile_against_train_distribution(score_test, score_train).to_numpy()
    pred[CONFIDENCE_COL] = ((pred[SIGNAL_SCORE_COL] - 0.5).abs() * 2).clip(0.01, 1.0)
    pred[DIRECTION_COL] = np.where(pred[SIGNAL_SCORE_COL] >= config['long_signal_threshold'], 'long', 'no_trade')

    # Keep explicit simulator/export aliases.
    pred['model_name'] = pred['model']
    pred['prediction_score'] = pred[SCORE_COL]
    pred['actual_value_for_ranking'] = make_ranking_target_values(pred[TRUE_COL], target_name).to_numpy()

    metrics = focused_metrics_for_prediction_frame(pred, target_cfg, target_name, pct=config['top_bottom_pct'])
    metrics.update({
        'ticker': ticker,
        'feature_set': config['feature_set_name'],
        'model': model_name,
        'walkforward_scheme': fold['walkforward_scheme'],
        'window_label': fold['window_label'],
        'fold_id': fold['fold_id'],
        'train_start_year': fold['train_start_year'],
        'train_end_year': fold['train_end_year'],
        'n_train_years': fold['n_train_years'],
        'test_year': fold['test_year'],
        'n_train': len(y_train),
        'n_test': len(y_test),
        'score_source': 'regression_prediction' if task == 'regression' else 'positive_or_long_probability',
    })
    return metrics, pred, {'status': 'completed', **fold, 'n_train': len(y_train), 'n_test': len(y_test)}


def run_focused_rolling_single_ticker_experiments(frames, all_data, feature_sets, target_configs=TARGET_CONFIGS, config=ROLLING_SINGLE_TICKER_CONFIG):
    """Main focused runner."""
    base = _ensure_rolling_base_data(frames, all_data)
    available_targets = available_target_configs(base, target_configs)

    if config['feature_set_name'] not in feature_sets:
        raise ValueError(f"Feature set {config['feature_set_name']} not found. Available: {list(feature_sets.keys())[:20]}")
    feature_cols = [c for c in feature_sets[config['feature_set_name']] if c in base.columns]
    if len(feature_cols) == 0:
        raise ValueError('No usable feature columns found for the selected feature set.')

    fold_rows = []
    metric_rows = []
    prediction_parts = []
    skipped_rows = []

    for ticker in config['tickers']:
        folds = build_single_ticker_rolling_folds(
            base,
            ticker=ticker,
            train_window_years=config['train_window_years'],
            n_test_folds=config['n_test_folds'],
            test_years=config.get('test_years'),
        )
        fold_rows.extend(folds)

        for target_name in config['target_names']:
            if target_name not in available_targets:
                skipped_rows.append({'ticker': ticker, 'target_name': target_name, 'status': 'skipped', 'reason': 'target_unavailable'})
                continue
            target_cfg = available_targets[target_name]
            models = get_models_for_task(target_cfg['task'])
            selected_model_names = get_focused_model_names_for_task(target_cfg['task'], config)

            for model_name in selected_model_names:
                if model_name not in models:
                    skipped_rows.append({'ticker': ticker, 'target_name': target_name, 'model': model_name, 'status': 'skipped', 'reason': 'model_unavailable'})
                    continue
                model = models[model_name]

                for fold in folds:
                    print(f"Rolling single ticker | ticker={ticker} | target={target_name} | model={model_name} | train={fold['train_start_year']}-{fold['train_end_year']} | test={fold['test_year']}")
                    try:
                        metrics, pred, status = fit_predict_focused_single_ticker_fold(
                            base, fold, target_name, target_cfg, feature_cols, model_name, model, config
                        )
                        if metrics is not None:
                            metric_rows.append(metrics)
                        if pred is not None and len(pred):
                            prediction_parts.append(pred)
                        if status.get('status') != 'completed':
                            status.update({'target_name': target_name, 'model': model_name})
                            skipped_rows.append(status)
                    except Exception as e:
                        skipped_rows.append({'ticker': ticker, 'target_name': target_name, 'model': model_name, 'fold_id': fold['fold_id'], 'status': 'error', 'reason': str(e)})
                        print('  ERROR:', e)

    fold_df = pd.DataFrame(fold_rows)
    metrics_by_fold = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False) if prediction_parts else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows)

    # Overall summary across the three test years for each ticker-target-model.
    overall_rows = []
    if len(predictions):
        group_cols = ['ticker', 'target_name', 'task', 'feature_set', 'model', 'walkforward_scheme']
        for keys, g in predictions.groupby(group_cols, dropna=False):
            target_name = keys[1]
            target_cfg = target_configs.get(target_name, {'task': g['task'].iloc[0]})
            row = focused_metrics_for_prediction_frame(g, target_cfg, target_name, pct=config['top_bottom_pct'])
            for col, value in zip(group_cols, keys):
                row[col] = value
            row['n_folds'] = g['fold_id'].nunique() if 'fold_id' in g.columns else np.nan
            row['test_years'] = ','.join(map(str, sorted(g['test_year'].dropna().astype(int).unique()))) if 'test_year' in g.columns else ''
            overall_rows.append(row)
    overall_metrics = pd.DataFrame(overall_rows)

    # Select clean export columns for simulator-style row-level predictions.
    export_cols = [
        c for c in [
            INDEX_COL, DATE_COL, YEAR_COL, TICKER_COL, SIC2_COL,
            'ticker', 'target_name', 'task', 'feature_set', 'model_name', 'model',
            'walkforward_scheme', 'window_label', 'fold_id', 'train_start_year', 'train_end_year', 'test_year',
            TRUE_COL, PRED_COL, SCORE_COL, 'prediction_score', SIGNAL_SCORE_COL, DIRECTION_COL, CONFIDENCE_COL,
            'actual_value_for_ranking', SPLIT_COL,
        ] if c in predictions.columns
    ]
    prediction_export = predictions[export_cols].copy() if len(predictions) else pd.DataFrame()

    # Register tables if the notebook export manager is available.
    try:
        register_table('focused_rolling_single_ticker_folds', fold_df)
        register_table('focused_rolling_single_ticker_metrics_by_fold', metrics_by_fold)
        register_table('focused_rolling_single_ticker_metrics_overall', overall_metrics)
        register_table('focused_rolling_single_ticker_skipped', skipped)
    except Exception:
        pass

    # Save files immediately.
    out_dir = OUTPUT_DIR / 'focused_rolling_single_ticker'
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_fold.to_csv(out_dir / 'focused_rolling_metrics_by_fold.csv', index=False)
    overall_metrics.to_csv(out_dir / 'focused_rolling_metrics_overall.csv', index=False)
    skipped.to_csv(out_dir / 'focused_rolling_skipped.csv', index=False)
    fold_df.to_csv(out_dir / 'focused_rolling_folds.csv', index=False)
    if len(prediction_export):
        prediction_export.to_csv(out_dir / 'focused_rolling_row_level_predictions.csv', index=False)
        prediction_export.to_parquet(out_dir / 'focused_rolling_row_level_predictions.parquet', index=False)

    # One Excel workbook for easier inspection.
    excel_path = out_dir / 'focused_rolling_single_ticker_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        fold_df.to_excel(writer, sheet_name='folds', index=False)
        metrics_by_fold.to_excel(writer, sheet_name='metrics_by_fold', index=False)
        overall_metrics.to_excel(writer, sheet_name='metrics_overall', index=False)
        skipped.to_excel(writer, sheet_name='skipped', index=False)
        # Excel row limit protection: write a preview only.
        prediction_export.head(50000).to_excel(writer, sheet_name='prediction_preview', index=False)

    print('Saved focused rolling outputs to:', out_dir.resolve())
    print('Excel summary:', excel_path.resolve())
    return metrics_by_fold, overall_metrics, prediction_export, skipped, fold_df


# ============================================================
# Run AAPL first
# ============================================================
# If frames/all_data/FEATURE_SET_CONFIGS are not already created in your session,
# run the notebook's data-loading and feature-set cells first:
# frames, all_data = load_dataset_from_config(DATASET_CONFIG)
# frames, all_data = apply_target_construction(frames)
# FEATURE_SET_CONFIGS = build_feature_sets(all_data, DATASET_CONFIG)

focused_metrics_by_fold, focused_metrics_overall, focused_predictions, focused_skipped, focused_folds = run_focused_rolling_single_ticker_experiments(
    frames=frames,
    all_data=all_data,
    feature_sets=FEATURE_SET_CONFIGS,
    target_configs=TARGET_CONFIGS,
    config=ROLLING_SINGLE_TICKER_CONFIG,
)

# Quick display
try:
    display(focused_metrics_overall.sort_values(['target_name', 'model']).head(30))
    display(focused_metrics_by_fold.sort_values(['target_name', 'model', 'test_year']).head(50))
    display(focused_predictions.head(20))
except Exception:
    print(focused_metrics_overall.head(30))
    print(focused_metrics_by_fold.head(50))
    print(focused_predictions.head(20))

# ============================================================
# Later robustness extension: PG / NVDA / BAC
# ============================================================
# ROLLING_SINGLE_TICKER_CONFIG_PG_NVDA_BAC = ROLLING_SINGLE_TICKER_CONFIG.copy()
# ROLLING_SINGLE_TICKER_CONFIG_PG_NVDA_BAC['tickers'] = ['PG', 'NVDA', 'BAC']
# focused_metrics_by_fold_extra, focused_metrics_overall_extra, focused_predictions_extra, focused_skipped_extra, focused_folds_extra = run_focused_rolling_single_ticker_experiments(
#     frames=frames,
#     all_data=all_data,
#     feature_sets=FEATURE_SET_CONFIGS,
#     target_configs=TARGET_CONFIGS,
#     config=ROLLING_SINGLE_TICKER_CONFIG_PG_NVDA_BAC,
# )
