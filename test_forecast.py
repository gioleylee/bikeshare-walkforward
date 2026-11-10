from datetime import date, timedelta
from unittest import TestCase

import numpy as np

from forecast import features, intervals, rolling_predict


class ForecastTests(TestCase):
    def test_features_never_read_same_day_or_future_counts(self):
        dates = [date(2011, 1, 1) + timedelta(days=n) for n in range(50)]
        counts = np.arange(50)
        holidays = np.zeros(50)
        amended = counts.copy()
        amended[35:] += 100
        clean = features(dates, counts, holidays)
        corrupted = features(dates, amended, holidays)
        np.testing.assert_array_equal(clean[:8], corrupted[:8])
        self.assertFalse(np.array_equal(clean[8], corrupted[8]))

    def test_refit_never_sees_labels_from_prediction_window(self):
        class LatestLabel:
            def fit(self, x, y):
                self.latest = y[-1]

            def predict(self, x):
                return np.repeat(self.latest, len(x))

        y = np.arange(10)
        np.testing.assert_array_equal(rolling_predict(LatestLabel(), y, y, 3, 2), [2, 2, 4, 4, 6, 6, 8])

    def test_intervals_use_only_past_forecast_errors(self):
        pred = np.repeat(50, 5)
        truth = np.array([50, 53, 57, 200, 60])
        lower, upper = intervals(pred, truth, window=3, coverage=0.9)
        self.assertTrue(np.isnan(lower[:3]).all())
        self.assertEqual((lower[3], upper[3]), (43, 57))
        self.assertEqual((lower[4], upper[4]), (0, 200))
