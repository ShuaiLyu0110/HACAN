import unittest

import torch

from meter.modules.hacan import (
    global_contrastive_divergence,
    hacan_similarity,
    key_semantic_filter_score,
)


class HACANAlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.images_h = torch.randn(3, 4, 8, requires_grad=True)
        self.images_m = torch.randn(3, 6, 8, requires_grad=True)
        self.texts_h = torch.randn(3, 7, 8, requires_grad=True)
        self.texts_m = torch.randn(3, 7, 8, requires_grad=True)
        self.lengths = [7, 6, 5]

    def test_similarity_is_finite_and_differentiable(self):
        for direction in ("i2t", "t2i"):
            scores = hacan_similarity(
                self.images_h,
                self.texts_h,
                self.images_m,
                self.texts_m,
                self.lengths,
                direction,
            )
            self.assertEqual(tuple(scores.shape), (3, 3))
            self.assertTrue(torch.isfinite(scores).all())
            scores.mean().backward(retain_graph=True)
        self.assertTrue(torch.isfinite(self.images_h.grad).all())

    def test_padding_does_not_change_ksf_score(self):
        scores = key_semantic_filter_score(
            self.images_h, self.texts_h, self.lengths, "i2t"
        )
        changed = self.texts_h.detach().clone()
        changed[2, 5:] = 1000
        changed_scores = key_semantic_filter_score(
            self.images_h, changed, self.lengths, "i2t"
        )
        torch.testing.assert_close(scores[:, 2], changed_scores[:, 2])

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            key_semantic_filter_score(
                self.images_h, self.texts_h, self.lengths, "both"
            )

    def test_gcd_loss_is_finite_and_differentiable(self):
        images = torch.randn(4, 8, requires_grad=True)
        captions = torch.randn(4, 8, requires_grad=True)
        text_negatives = torch.tensor([1, 2, 3, 0])
        image_negatives = torch.tensor([2, 3, 0, 1])
        loss = global_contrastive_divergence(
            images, captions, text_negatives, image_negatives
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(images.grad).all())


if __name__ == "__main__":
    unittest.main()
