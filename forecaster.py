import vars
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# ---------- metrics ----------
def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def smape(y, yhat, eps=1e-8):
    num = np.abs(y - yhat)
    den = (np.abs(y) + np.abs(yhat) + eps) / 2.0
    return float(100 * np.mean(num / den))  # percent

def mase(y_true, y_pred, y_train, m=12):
    # scale by seasonal naive MAE on the *training* period
    if len(y_train) <= m:
        return np.nan
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if denom == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)

# ---------- month normalization ----------
def _normalize_month_index(df):
    d = df.copy()
    m = d['month']
    if pd.api.types.is_period_dtype(m):
        d['month'] = pd.PeriodIndex(m, freq='M').to_timestamp()
    elif pd.api.types.is_datetime64_any_dtype(m):
        d['month'] = m.dt.to_period('M').dt.to_timestamp()
    else:
        d['month'] = pd.to_datetime(m, errors='coerce').dt.to_period('M').dt.to_timestamp()
    d = d.sort_values('month').set_index('month')
    return d

# Fourier-based transformer function
def forecast_12_fourier(outcome_trend, horizon=12, K=3, period=12):
    df = outcome_trend.copy()

    # --- normalize month to Timestamp at month-start ---
    m = df['month']
    if pd.api.types.is_period_dtype(m):
        # Coerce to monthly periods and convert to timestamps (start of month)
        df['month'] = pd.PeriodIndex(m, freq='M').to_timestamp()  # <-- no 'MS' here
    elif pd.api.types.is_datetime64_any_dtype(m):
        df['month'] = m.dt.to_period('M').dt.to_timestamp()
    else:
        df['month'] = pd.to_datetime(m, errors='coerce').dt.to_period('M').dt.to_timestamp()

    df = df.sort_values('month').reset_index(drop=True)

    # time index
    t = np.arange(len(df))
    t_future = np.arange(len(df), len(df) + horizon)

    def fourier_basis(tt, K, period):
        X = [np.ones_like(tt), tt]  # intercept + linear trend
        for k in range(1, K + 1):
            X.append(np.sin(2 * np.pi * k * tt / period))
            X.append(np.cos(2 * np.pi * k * tt / period))
        return np.column_stack(X)

    X  = fourier_basis(t, K, period)
    Xf = fourier_basis(t_future, K, period)

    target_cols = vars.out_names
    preds = {}

    for col in target_cols:
        y = df[col].astype(float).to_numpy()
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        yhat = model.predict(Xf)
        if col == 'days_on_market':
            yhat = np.clip(yhat, 0, None)
        preds[col] = yhat

    future_months = pd.date_range(df['month'].iloc[-1] + pd.offsets.MonthBegin(1),
                                  periods=horizon, freq='MS')
    out = pd.DataFrame({'month': future_months})
    for col in target_cols:
        out[col] = preds[col]
    return out

# ---------- backtester ----------
def backtest_forecaster(outcome_trend, forecaster, horizon=12, seasonality=12, min_train=36, step=1):
    """
    forecaster: function(train_df, horizon) -> DataFrame with 'month' and same numeric columns
    """
    df = _normalize_month_index(outcome_trend)
    cols = [c for c in df.columns if c != 'month']  # index now holds month
    n = len(df)
    cut_starts = list(range(min_train-1, n - horizon, step))  # index of last training obs

    # collectors
    per_h_metrics = {c: {h+1: [] for h in range(horizon)} for c in cols}
    overall = {c: {"RMSE": [], "sMAPE": [], "MASE": [], "RMSE_baseline": [], "sMAPE_baseline": [], "MASE_baseline": []} for c in cols}

    for cut in cut_starts:
        train = df.iloc[:cut+1].copy()
        future_idx = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')

        # model forecast
        fcast = forecaster(train.reset_index().rename(columns={'index': 'month'}), horizon=horizon)
        fcast = _normalize_month_index(fcast).reindex(future_idx)

        # actuals
        truth = df.iloc[cut+1:cut+1+horizon].reindex(future_idx)

        # seasonal naive baseline: y_{t+h} = y_{t+h-12}
        baseline = df[cols].shift(seasonality).iloc[cut+1:cut+1+horizon].reindex(future_idx)

        # compute metrics per column
        for c in cols:
            y_true = truth[c].astype(float).to_numpy()
            y_hat  = fcast[c].astype(float).to_numpy()
            y_base = baseline[c].astype(float).to_numpy()

            mask_model = ~np.isnan(y_true) & ~np.isnan(y_hat)
            mask_base  = ~np.isnan(y_true) & ~np.isnan(y_base)

            if mask_model.any():
                overall[c]["RMSE"].append(rmse(y_true[mask_model], y_hat[mask_model]))
                overall[c]["sMAPE"].append(smape(y_true[mask_model], y_hat[mask_model]))
                # MASE needs training history of this column
                overall[c]["MASE"].append(mase(y_true[mask_model], y_hat[mask_model], train[c].astype(float).to_numpy(), m=seasonality))
                # per-horizon
                for h in range(horizon):
                    if not np.isnan(y_true[h]) and not np.isnan(y_hat[h]):
                        per_h_metrics[c][h+1].append(abs(y_true[h]-y_hat[h]))

            if mask_base.any():
                overall[c]["RMSE_baseline"].append(rmse(y_true[mask_base], y_base[mask_base]))
                overall[c]["sMAPE_baseline"].append(smape(y_true[mask_base], y_base[mask_base]))
                overall[c]["MASE_baseline"].append(mase(y_true[mask_base], y_base[mask_base], train[c].astype(float).to_numpy(), m=seasonality))

    # aggregate
    rows = []
    for c in cols:
        if overall[c]["RMSE"]:
            rmse_m   = np.mean(overall[c]["RMSE"])
            rmse_b   = np.mean(overall[c]["RMSE_baseline"]) if overall[c]["RMSE_baseline"] else np.nan
            smape_m  = np.mean(overall[c]["sMAPE"])
            smape_b  = np.mean(overall[c]["sMAPE_baseline"]) if overall[c]["sMAPE_baseline"] else np.nan
            mase_m   = np.nanmean(overall[c]["MASE"])
            mase_b   = np.nanmean(overall[c]["MASE_baseline"]) if overall[c]["MASE_baseline"] else np.nan
            rows.append({
                "series": c,
                "RMSE": rmse_m,
                "RMSE_baseline": rmse_b,
                "RMSE_improvement_%": (1 - rmse_m/rmse_b)*100 if rmse_b and rmse_b>0 else np.nan,
                "sMAPE_%": smape_m,
                "sMAPE_baseline_%": smape_b,
                "sMAPE_improvement_%": (1 - smape_m/smape_b)*100 if smape_b and smape_b>0 else np.nan,
                "MASE": mase_m,               # <1 means better than seasonal naive
                "MASE_baseline": mase_b       # ~1 for seasonal naive
            })

    summary = pd.DataFrame(rows).sort_values("series")

    # Optional: mean absolute error by forecast horizon (useful to see decay)
    by_horizon = {}
    for c in cols:
        by_horizon[c] = pd.DataFrame({
            "horizon": list(per_h_metrics[c].keys()),
            "mean_abs_error": [np.mean(per_h_metrics[c][h]) if per_h_metrics[c][h] else np.nan
                               for h in per_h_metrics[c]]
        })

    return summary, by_horizon

# your forecaster (wraps the function you already have)
def fourier_forecaster(train_df, horizon):
    return forecast_12_fourier(train_df, horizon=horizon, K=3, period=12)