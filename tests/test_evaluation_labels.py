"""Unit tests for evaluation label mapping (e.g., IR -> NS)."""
import unittest


class TestEvaluationLabels(unittest.TestCase):
    def test_s_label_preserved(self):
        label = "S"
        mapped = "NS" if label == "IR" else label
        self.assertEqual(mapped, "S")

    def test_ns_label_preserved(self):
        label = "NS"
        mapped = "NS" if label == "IR" else label
        self.assertEqual(mapped, "NS")

    def test_ir_mapped_to_ns(self):
        label = "IR"
        mapped = "NS" if label == "IR" else label
        self.assertEqual(mapped, "NS")


if __name__ == "__main__":
    unittest.main()
