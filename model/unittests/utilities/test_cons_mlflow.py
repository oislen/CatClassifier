import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cons


class TestMlflowRegisteredModelNames(unittest.TestCase):
    """Unit tests for cons.mlflow_registered_model_names."""

    def setUp(self):
        self.names = cons.mlflow_registered_model_names
        self.expected_keys = [
            "alexnet_8_pretrained",
            "vgg_16_pretrained",
            "resnet_50_pretrained",
        ]

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_registered_model_names_is_dict(self):
        self.assertIsInstance(self.names, dict)

    def test_registered_model_names_values_are_strings(self):
        for v in self.names.values():
            self.assertIsInstance(v, str)
            self.assertGreater(len(v), 0)

    # ------------------------------------------------------------------
    # Key tests
    # ------------------------------------------------------------------

    def test_registered_model_names_contains_all_architectures(self):
        for key in self.expected_keys:
            self.assertIn(key, self.names)

    def test_registered_model_names_key_count(self):
        self.assertEqual(len(self.names), len(self.expected_keys))

    # ------------------------------------------------------------------
    # Naming convention tests
    # ------------------------------------------------------------------

    def test_registered_model_names_follow_naming_convention(self):
        """All registered names must start with 'CatClassifier-'."""
        for v in self.names.values():
            self.assertTrue(
                v.startswith("CatClassifier-"),
                msg=f"Registered name {v!r} does not start with 'CatClassifier-'",
            )

    def test_registered_model_names_no_spaces(self):
        """MLflow rejects names with leading or trailing spaces."""
        for v in self.names.values():
            self.assertEqual(v, v.strip(), msg=f"Name {v!r} has leading/trailing whitespace")

    # ------------------------------------------------------------------
    # Exact value tests
    # ------------------------------------------------------------------

    def test_alexnet_registered_name(self):
        self.assertEqual(self.names["alexnet_8_pretrained"], "CatClassifier-AlexNet8")

    def test_vgg16_registered_name(self):
        self.assertEqual(self.names["vgg_16_pretrained"], "CatClassifier-VGG16")

    def test_resnet50_registered_name(self):
        self.assertEqual(self.names["resnet_50_pretrained"], "CatClassifier-ResNet50")


if __name__ == "__main__":
    unittest.main()
