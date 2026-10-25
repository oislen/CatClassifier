import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch_models.CustomDataset import CustomDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n, h=32, w=32, c=3, seed=0):
    """Build a minimal DataFrame matching TorchLoadImages output."""
    torch.manual_seed(seed)
    return pd.DataFrame(
        {
            "image_tensors":    [torch.rand(c, h, w) for _ in range(n)],
            "category_tensors": [torch.tensor(i % 2, dtype=torch.int64) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

image_h, image_w, image_c = 32, 32, 3
n_samples = 5
df_5       = _make_df(n_samples)
df_single  = _make_df(1)
df_large   = _make_df(20)

# Float-type category tensors
df_float_cat = pd.DataFrame(
    {
        "image_tensors":    [torch.rand(image_c, image_h, image_w)],
        "category_tensors": [torch.tensor(0.0)],
    }
)

obs_dataset        = CustomDataset(df_5)
obs_dataset_single = CustomDataset(df_single)
obs_dataset_large  = CustomDataset(df_large)
obs_dataset_float  = CustomDataset(df_float_cat)


class TestCustomDataset(unittest.TestCase):
    """
    Unit tests for CustomDataset.

    Verifies __len__, __getitem__, collate_fn, DataLoader integration,
    single-item and large datasets, and float image tensor inputs.
    """

    def setUp(self):
        self.dataset        = obs_dataset
        self.dataset_single = obs_dataset_single
        self.dataset_large  = obs_dataset_large
        self.n_samples      = n_samples

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type(self):
        from torch.utils.data import Dataset
        self.assertIsInstance(self.dataset, Dataset)

    # ------------------------------------------------------------------
    # Length tests
    # ------------------------------------------------------------------

    def test_len(self):
        self.assertEqual(len(self.dataset), self.n_samples)

    def test_len_single(self):
        self.assertEqual(len(self.dataset_single), 1)

    def test_len_large(self):
        self.assertEqual(len(obs_dataset_large), 20)

    # ------------------------------------------------------------------
    # __getitem__ tests
    # ------------------------------------------------------------------

    def test_getitem_returns_tuple(self):
        item = self.dataset[0]
        self.assertIsInstance(item, tuple)

    def test_getitem_tuple_length(self):
        item = self.dataset[0]
        self.assertEqual(len(item), 2)

    def test_getitem_image_tensor_type(self):
        img, _ = self.dataset[0]
        self.assertIsInstance(img, torch.Tensor)

    def test_getitem_category_tensor_type(self):
        _, cat = self.dataset[0]
        self.assertIsInstance(cat, torch.Tensor)

    def test_getitem_image_tensor_shape(self):
        img, _ = self.dataset[0]
        self.assertEqual(img.shape, (image_c, image_h, image_w))

    def test_getitem_last_index(self):
        """Access the final element without IndexError."""
        item = self.dataset[self.n_samples - 1]
        self.assertEqual(len(item), 2)

    def test_getitem_all_indices(self):
        """Every valid index should return a (image, category) tuple."""
        for i in range(self.n_samples):
            item = self.dataset[i]
            self.assertEqual(len(item), 2)

    def test_getitem_category_values(self):
        """Categories should round-trip correctly from the DataFrame."""
        for i in range(self.n_samples):
            _, cat = self.dataset[i]
            self.assertEqual(cat.item(), i % 2)

    def test_getitem_single_dataset(self):
        img, cat = self.dataset_single[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(cat, torch.Tensor)

    # ------------------------------------------------------------------
    # Image tensor value tests
    # ------------------------------------------------------------------

    def test_image_tensor_values_are_finite(self):
        for i in range(self.n_samples):
            img, _ = self.dataset[i]
            self.assertTrue(torch.isfinite(img).all())

    def test_image_tensor_dtype(self):
        img, _ = self.dataset[0]
        self.assertEqual(img.dtype, torch.float32)

    def test_category_tensor_dtype(self):
        _, cat = self.dataset[0]
        self.assertEqual(cat.dtype, torch.int64)

    # ------------------------------------------------------------------
    # collate_fn tests
    # ------------------------------------------------------------------

    def test_collate_fn_returns_tuple(self):
        batch = [self.dataset[i] for i in range(3)]
        result = CustomDataset.collate_fn(batch)
        self.assertIsInstance(result, tuple)

    def test_collate_fn_tuple_length(self):
        batch = [self.dataset[i] for i in range(3)]
        result = CustomDataset.collate_fn(batch)
        self.assertEqual(len(result), 2)

    def test_collate_fn_image_batch_shape(self):
        batch_size = 3
        batch = [self.dataset[i] for i in range(batch_size)]
        images, _ = CustomDataset.collate_fn(batch)
        self.assertEqual(images.shape, (batch_size, image_c, image_h, image_w))

    def test_collate_fn_category_batch_shape(self):
        batch_size = 3
        batch = [self.dataset[i] for i in range(batch_size)]
        _, categories = CustomDataset.collate_fn(batch)
        self.assertEqual(categories.shape, (batch_size,))

    def test_collate_fn_image_tensor_type(self):
        batch = [self.dataset[i] for i in range(2)]
        images, _ = CustomDataset.collate_fn(batch)
        self.assertIsInstance(images, torch.Tensor)

    def test_collate_fn_category_tensor_type(self):
        batch = [self.dataset[i] for i in range(2)]
        _, categories = CustomDataset.collate_fn(batch)
        self.assertIsInstance(categories, torch.Tensor)

    def test_collate_fn_single_item_batch(self):
        """collate_fn with a one-item list should still produce correct shapes."""
        batch = [self.dataset[0]]
        images, categories = CustomDataset.collate_fn(batch)
        self.assertEqual(images.shape, (1, image_c, image_h, image_w))
        self.assertEqual(categories.shape, (1,))

    def test_collate_fn_full_batch(self):
        """collate_fn over the whole dataset should produce correct shapes."""
        batch = [self.dataset[i] for i in range(self.n_samples)]
        images, categories = CustomDataset.collate_fn(batch)
        self.assertEqual(images.shape[0], self.n_samples)
        self.assertEqual(categories.shape[0], self.n_samples)

    # ------------------------------------------------------------------
    # DataLoader integration tests
    # ------------------------------------------------------------------

    def test_dataloader_batch_shape(self):
        """A DataLoader with batch_size=2 should yield correctly shaped batches."""
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        images, categories = next(iter(loader))
        self.assertEqual(images.shape, (2, image_c, image_h, image_w))
        self.assertEqual(categories.shape, (2,))

    def test_dataloader_total_samples(self):
        """DataLoader must iterate over all samples exactly once."""
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        total = sum(imgs.shape[0] for imgs, _ in loader)
        self.assertEqual(total, self.n_samples)

    def test_dataloader_with_collate_fn(self):
        """Custom collate_fn should work seamlessly with DataLoader."""
        # batch_size must be a positive integer when collate_fn is supplied;
        # batch_size=None bypasses collation entirely (passes single items).
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=CustomDataset.collate_fn,
        )
        for images, categories in loader:
            self.assertIsInstance(images, torch.Tensor)
            self.assertIsInstance(categories, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
