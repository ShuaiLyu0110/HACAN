import unittest

import numpy as np

from meter.modules.eval_gl import i2t_gl, t2i_gl


def zero_similarity(images, captions, lengths, module):
    return np.zeros((len(images), len(captions)), dtype=np.float32)


class HierarchicalInferenceTest(unittest.TestCase):
    def setUp(self):
        image_global = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.full_images = np.repeat(image_global, 5, axis=0)
        self.full_captions = np.repeat(image_global, 5, axis=0)
        self.images_h = np.zeros((10, 2, 3), dtype=np.float32)
        self.images_m = np.zeros((10, 3, 3), dtype=np.float32)
        self.captions_h = np.zeros((10, 4, 3), dtype=np.float32)
        self.captions_m = np.zeros((10, 4, 3), dtype=np.float32)
        self.lengths = np.full(10, 4)

    def test_topk_then_rerank_retrieves_correct_pairs(self):
        i2t_metrics, _, coarse = i2t_gl(
            self.full_images,
            self.full_captions,
            self.images_h,
            self.captions_h,
            self.images_m,
            self.captions_m,
            None,
            self.lengths,
            sim_function=zero_similarity,
            sim_function_new=zero_similarity,
            topk=5,
        )
        t2i_metrics, _ = t2i_gl(
            self.full_images,
            self.full_captions,
            self.images_h,
            self.captions_h,
            self.images_m,
            self.captions_m,
            None,
            self.lengths,
            sim_function=zero_similarity,
            sim_function_new=zero_similarity,
            sims=coarse,
            topk=1,
        )
        self.assertEqual(i2t_metrics[0], 100.0)
        self.assertEqual(t2i_metrics[0], 100.0)

    def test_missing_coarse_candidate_is_not_counted_as_rank_one(self):
        wrong_captions = self.full_captions.copy()
        wrong_captions[:5] = [0.0, 1.0]
        metrics, _, _ = i2t_gl(
            self.full_images,
            wrong_captions,
            self.images_h,
            self.captions_h,
            self.images_m,
            self.captions_m,
            None,
            self.lengths,
            sim_function=zero_similarity,
            sim_function_new=zero_similarity,
            topk=1,
        )
        self.assertLess(metrics[0], 100.0)


if __name__ == "__main__":
    unittest.main()
