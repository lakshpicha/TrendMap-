import unittest
import pandas as pd
import numpy as np

from helpers import (
    compute_trend_score,
    spike_alerts,
    auto_insights,
    predict_lstm,
    chatbot_reply,
    to_excel
)
from utils.validator import validate_keywords, validate_geo

class TestAutomation(unittest.TestCase):

    def test_compute_trend_score(self):
        # Empty series
        score, label, delta = compute_trend_score(pd.Series(dtype=float))
        self.assertEqual(score, 50)
        self.assertEqual(label, "➡️ Stable")
        self.assertEqual(delta, 0.0)

        # Growing series
        s = pd.Series([10, 10, 10, 10, 20, 20, 20, 20])
        score, label, delta = compute_trend_score(s)
        self.assertTrue(score > 50)
        self.assertEqual(label, "📈 Growing")
        self.assertEqual(delta, 100.0)

    def test_spike_alerts(self):
        data = pd.DataFrame({
            'kw1': [10, 10, 10, 100, 10, 10], # Spike
            'kw2': [20, 20, 20, 20, 20, 20]   # No spike
        }, index=pd.date_range("2023-01-01", periods=6))
        
        alerts = spike_alerts(data, ['kw1', 'kw2'])
        self.assertEqual(len(alerts), 1)
        self.assertIn("kw1", alerts[0])
        
    def test_auto_insights(self):
        data = pd.DataFrame({
            'kw1': [10, 20, 30, 40, 50, 60]
        }, index=pd.date_range("2023-01-01", periods=6))
        
        insight = auto_insights(data, ['kw1', 'missing_kw'])
        self.assertIn("kw1", insight)
        self.assertNotIn("missing_kw", insight)

    def test_predict_lstm_fallback(self):
        # Even if TF is available, let's test predict_lstm with small data
        data = pd.DataFrame({'kw1': [10, 15, 20]}, index=pd.date_range("2023-01-01", periods=3))
        res = predict_lstm(data)
        self.assertEqual(len(res), 16) # 1 bridge + 15 predicted

    def test_predict_lstm_full(self):
        # 20 periods should trigger real LSTM if TF available
        data = pd.DataFrame({'kw1': np.linspace(10, 100, 20)}, index=pd.date_range("2023-01-01", periods=20))
        res = predict_lstm(data)
        self.assertEqual(len(res), 16)

    def test_chatbot_reply(self):
        data = pd.DataFrame({
            'kw1': [10, 20, 30]
        }, index=pd.date_range("2023-01-01", periods=3))
        
        # Trend question
        rep = chatbot_reply("trend kw1", data, ['kw1'])
        self.assertIn("kw1", rep)

        # Peak question
        rep = chatbot_reply("when was the peak?", data, ['kw1'])
        self.assertIn("peak search interest", rep)

    def test_validators(self):
        clean, err = validate_keywords("ai,  ML , ")
        self.assertEqual(clean, ["ai", "ML"])
        self.assertIsNone(err)

        clean, err = validate_geo("us")
        self.assertEqual(clean, "US")
        
        clean, err = validate_geo("XYZ")
        self.assertIsNotNone(err)

if __name__ == "__main__":
    unittest.main()
