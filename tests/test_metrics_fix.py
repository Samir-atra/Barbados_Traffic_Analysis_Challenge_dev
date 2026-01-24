
import unittest
import warnings
from sklearn.metrics import classification_report
import numpy as np

class TestMetricsFix(unittest.TestCase):
    def test_classification_report_zero_division(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # Class 1 is never predicted
        labels = ['class_0', 'class_1']
        
        # This triggers the warning if zero_division is not handled (default is 'warn')
        # We want to assert that passing zero_division=0 works and suppresses warning if we could catch it,
        # or just verify it runs without error.
        
        # Using catch_warnings to assert that no warning is raised when zero_division=0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Cause all warnings to always be triggered.
            report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
            
            # Check that no UndefinedMetricWarning was raised
            undefined_metric_warnings = [x for x in w if "UndefinedMetricWarning" in str(x.category)]
            self.assertEqual(len(undefined_metric_warnings), 0, "UndefinedMetricWarning should not be raised with zero_division=0")
            
        print("\nTest passed: classification_report handled zero_division correctly.")

if __name__ == '__main__':
    unittest.main()
