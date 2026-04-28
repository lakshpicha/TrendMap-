import pandas as pd
import numpy as np
import io
import logging

# Optional ML stack
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ------------------------------------------------------------------
# TREND SCORING
# ------------------------------------------------------------------
def compute_trend_score(series):
    """
    Returns (score: int 0-100, label: str, delta: float %)
    Compares the last 4 data points vs the first 4 data points.
    """
    if series is None or len(series) < 4:
        return 50, "➡️ Stable", 0.0
    recent  = series.iloc[-4:].mean()
    earlier = series.iloc[:4].mean()
    if earlier == 0:
        return 50, "➡️ Stable", 0.0
    delta = ((recent - earlier) / earlier) * 100
    if delta > 10:
        return min(int(50 + delta), 100), "📈 Growing", delta
    elif delta < -10:
        return max(int(50 + delta), 0), "📉 Declining", delta
    return 50, "➡️ Stable", delta


# ------------------------------------------------------------------
# SPIKE DETECTION
# ------------------------------------------------------------------
def spike_alerts(data, keywords):
    """
    Returns a list of alert strings for keywords that spiked
    above 1.5 standard deviations from the mean.
    """
    alerts = []
    for kw in keywords:
        if kw not in data.columns:
            continue
        s = data[kw]
        mean, std = s.mean(), s.std()
        spikes = s[s > mean + 1.5 * std]
        if not spikes.empty:
            alerts.append(
                f"🔔 **{kw}** spiked on **{spikes.idxmax().strftime('%b %Y')}** "
                f"(value: {int(spikes.max())}, avg: {int(mean)})"
            )
    return alerts


# ------------------------------------------------------------------
# AI AUTO-INSIGHTS
# ------------------------------------------------------------------
def auto_insights(data, keywords):
    """Generates a text summary of trends for all keywords."""
    lines = []
    for kw in keywords:
        if kw not in data.columns:
            continue
        score, label, delta = compute_trend_score(data[kw])
        peak_date = data[kw].idxmax()
        peak_val  = int(data[kw].max())
        lines.append(
            f"**{kw}** — {label} | Score: **{score}/100** | Δ {delta:+.1f}% | "
            f"Peak: {peak_val} on {peak_date.strftime('%b %Y')}"
        )
    return "\n\n".join(lines) if lines else "No insights available."


# ------------------------------------------------------------------
# LSTM FORECAST
# ------------------------------------------------------------------
def predict_lstm(data):
    """
    Trains a lightweight LSTM model on Google Trends data and
    forecasts the next 15 time-periods.
    Falls back to flat-line projection if TensorFlow is unavailable.
    """
    cols = [c for c in data.columns if c != 'isPartial']
    vals = data[cols].values

    last = data.index[-1]
    freq = pd.infer_freq(data.index) or 'W'

    # --- Fallback (no TF or too little data) ---
    if not TF_AVAILABLE or len(vals) < 15:
        future_df = pd.DataFrame(
            [vals[-1]] * 15, columns=cols,
            index=pd.date_range(start=last, periods=16, freq=freq)[1:]
        )
        return pd.concat([data[cols].iloc[[-1]], future_df])

    # --- Real LSTM ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    lb = min(10, len(scaled) - 1)
    X, Y = [], []
    for i in range(len(scaled) - lb):
        X.append(scaled[i:i + lb])
        Y.append(scaled[i + lb])
    X, Y = np.array(X), np.array(Y)

    model = Sequential([
        LSTM(32, input_shape=(lb, vals.shape[1])),
        Dense(vals.shape[1])
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, Y, epochs=25, batch_size=2, verbose=0)

    last_batch = scaled[-lb:].reshape(1, lb, vals.shape[1])
    preds = []
    for _ in range(15):
        p = model.predict(last_batch, verbose=0)
        preds.append(p[0])
        last_batch = np.append(last_batch[:, 1:, :], [p], axis=1)

    unscaled = scaler.inverse_transform(preds)
    fidx     = pd.date_range(start=last, periods=16, freq=freq)[1:]
    fdf      = pd.DataFrame(unscaled, index=fidx, columns=cols)
    bridge   = pd.DataFrame([vals[-1]], index=[last], columns=cols)
    return pd.concat([bridge, fdf])


# ------------------------------------------------------------------
# CHATBOT
# ------------------------------------------------------------------
def chatbot_reply(question, data, keywords):
    """
    Rule-based AI analyst chatbot.
    Understands questions about trends, peaks, spikes, comparisons,
    forecasts, and auto-insights.
    """
    if not keywords or data is None or data.empty:
        return "⚠️ Please generate a dashboard first, then ask me about the data!"

    q  = question.lower()
    kw = keywords[0]

    if any(word in q for word in ["trend", "growing", "declining", "status"]):
        score, label, delta = compute_trend_score(data.get(kw, pd.Series()))
        return (f"**{kw}** is currently **{label}** with a trend score of **{score}/100** "
                f"and a momentum change of **{delta:+.1f}%** vs the start of the period.")

    if any(word in q for word in ["peak", "highest", "max", "top"]):
        if kw in data.columns:
            peak_date = data[kw].idxmax()
            peak_val  = int(data[kw].max())
            return (f"**{kw}** hit its peak search interest of **{peak_val}/100** "
                    f"on **{peak_date.strftime('%B %Y')}**.")

    if any(word in q for word in ["spike", "alert", "jump"]):
        alerts = spike_alerts(data, keywords)
        return "\n\n".join(alerts) if alerts else f"No significant spikes detected for **{kw}**."

    if any(word in q for word in ["compare", "vs", "versus", "difference"]):
        if len(keywords) > 1:
            avgs   = {k: round(data[k].mean(), 1) for k in keywords if k in data.columns}
            winner = max(avgs, key=avgs.get)
            lines  = " | ".join([f"**{k}**: avg {v}" for k, v in avgs.items()])
            return f"Comparison — {lines}.\n\n🏆 **{winner}** has the highest average interest."
        return "Enter multiple keywords (comma-separated) in the sidebar to compare."

    if any(word in q for word in ["why", "reason", "cause"]):
        return (f"**{kw}** interest is driven by media coverage, industry events, viral content, "
                f"and seasonal patterns. Peaks often align with product launches or news cycles.")

    if any(word in q for word in ["predict", "future", "forecast", "next"]):
        return (f"The LSTM model projects the next 15 time periods for **{kw}**. "
                f"Scroll up to view the dashed AI Forecast chart.")

    if any(word in q for word in ["summary", "insight", "tell me", "overview"]):
        return auto_insights(data, keywords)

    return (f"I can help with:\n"
            f"- *Is {kw} growing?*\n"
            f"- *When was the peak?*\n"
            f"- *Any spikes?*\n"
            f"- *Compare keywords*\n"
            f"- *Why is this trending?*\n"
            f"- *Predict future trends*\n"
            f"- *Give me the summary*")


# ------------------------------------------------------------------
# EXCEL EXPORT
# ------------------------------------------------------------------
def to_excel(dfs: dict) -> bytes:
    """Exports a dict of {sheet_name: DataFrame} to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:30])
    return output.getvalue()
