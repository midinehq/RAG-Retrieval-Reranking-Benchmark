"""Tests du contrat de l'application ; aucun artefact n'est réécrit."""
import unittest
import numpy as np
from rag_core import Pipeline, load_results

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline()
        cls.metrics, cls.traces, _ = load_results()

    def test_empty_question_rejected(self):
        for question in ("", "   ", None):
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.pipeline.all_scores(question)

    def test_unknown_words_have_no_lexical_signal(self):
        scores = self.pipeline.all_scores("zxqvvvzzzz")
        self.assertFalse(scores["_has_signal"])
        self.assertEqual(float(scores["bm25"].max()), 0.0)

    def test_saved_reader_is_binary_not_generative(self):
        self.assertFalse(self.metrics["settings"]["answers_are_generative"])
        decision, probability = self.pipeline.read("is the moon a planet", "The Moon is Earth's natural satellite.")
        self.assertIn(decision, ("Oui", "Non"))
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_selected_fusion_reduces_to_bm25(self):
        self.assertEqual(self.metrics["settings"]["fusion_alpha_bm25"], 1.0)
        scores = self.pipeline.all_scores(self.traces[0]["question"])
        np.testing.assert_allclose(scores["hybrid"], scores["bm25"] / (scores["bm25"].max() + 1e-9))

if __name__ == "__main__":
    unittest.main()