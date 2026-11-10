"""Unit tests for Part 1 (word overlap fact-checker).

Tests coverage:
- preprocess(): lowercasing, punctuation handling, stop-word removal,
  Porter stemming, and verifying each toggle when turned off.
- _compute_score(): recall, jaccard, f1, and rouge on small hand-made sets
  with known answers; empty fact returns 0.0; unknown scorer raises ValueError.
- predict(): threshold boundary behaviour (score >= threshold -> "S", < threshold -> "NS").
"""
import unittest

from factcheck import WordRecallThresholdFactChecker


class TestWordOverlapPreprocessing(unittest.TestCase):
    def setUp(self):
        self.default_checker = WordRecallThresholdFactChecker()

    def test_lowercasing(self):
        # Upper/mixed case words are converted to lowercase
        tokens = self.default_checker.preprocess("Ohio CLEVELAND")
        self.assertIn("ohio", tokens)
        self.assertIn("cleveland", tokens)

    def test_punctuation_split_off(self):
        # "Ohio." has punctuation stripped off, leaving just "ohio"
        tokens = self.default_checker.preprocess("Ohio.")
        self.assertEqual(tokens, {"ohio"})

    def test_stop_words_removed(self):
        # English stop words like "is", "an", "the" are excluded
        tokens = self.default_checker.preprocess("Marianne is an actress.")
        self.assertNotIn("is", tokens)
        self.assertNotIn("an", tokens)
        self.assertIn("actress", tokens)

    def test_stemming(self):
        # "inventing" should be stemmed to "invent"
        tokens = self.default_checker.preprocess("inventing")
        self.assertEqual(tokens, {"invent"})

    def test_toggle_use_nltk_off(self):
        # When use_nltk=False, uses text.split()
        checker = WordRecallThresholdFactChecker(use_nltk=False, remove_stopwords=False, use_stemming=False)
        tokens = checker.preprocess("Hello world")
        self.assertEqual(tokens, {"hello", "world"})

    def test_toggle_remove_stopwords_off(self):
        # When remove_stopwords=False, stopwords are retained
        checker = WordRecallThresholdFactChecker(remove_stopwords=False, use_stemming=False)
        tokens = checker.preprocess("This is a test.")
        self.assertIn("this", tokens)
        self.assertIn("is", tokens)

    def test_toggle_use_stemming_off(self):
        # When use_stemming=False, inflected word forms are preserved
        checker = WordRecallThresholdFactChecker(use_stemming=False)
        tokens = checker.preprocess("inventing")
        self.assertEqual(tokens, {"inventing"})


class TestWordOverlapScorers(unittest.TestCase):
    def test_recall_scorer(self):
        checker = WordRecallThresholdFactChecker(scorer="recall")
        fact_words = {"a", "b"}
        passage_words = {"b", "c", "d"}
        # Recall = |overlap| / |fact_words| = 1 / 2 = 0.5
        score = checker._compute_score(fact_words, passage_words, "a b", "b c d")
        self.assertAlmostEqual(score, 0.5)

    def test_jaccard_scorer(self):
        checker = WordRecallThresholdFactChecker(scorer="jaccard")
        fact_words = {"a", "b"}
        passage_words = {"b", "c", "d"}
        # Jaccard = |overlap| / |union| = 1 / 4 = 0.25
        score = checker._compute_score(fact_words, passage_words, "a b", "b c d")
        self.assertAlmostEqual(score, 0.25)

    def test_f1_scorer(self):
        checker = WordRecallThresholdFactChecker(scorer="f1")
        fact_words = {"a", "b"}
        passage_words = {"b", "c", "d"}
        # Precision = 1/3, Recall = 1/2 -> F1 = 2 * (1/3) * (1/2) / (1/3 + 1/2) = 0.4
        score = checker._compute_score(fact_words, passage_words, "a b", "b c d")
        self.assertAlmostEqual(score, 0.4)

    def test_rouge_scorer(self):
        checker = WordRecallThresholdFactChecker(scorer="rouge")
        # Raw text strings are used by ROUGE
        score = checker._compute_score(
            {"red", "blue"}, {"red", "green"},
            "red blue", "red green"
        )
        self.assertAlmostEqual(score, 0.5)

    def test_empty_fact_returns_zero(self):
        for scorer in ["recall", "jaccard", "f1", "rouge"]:
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                score = checker._compute_score(set(), {"apple", "banana"}, "", "apple banana")
                self.assertEqual(score, 0.0)

    def test_unknown_scorer_raises_value_error(self):
        checker = WordRecallThresholdFactChecker(scorer="invalid_scorer")
        with self.assertRaises(ValueError):
            checker._compute_score({"word"}, {"word"}, "word", "word")


class TestWordOverlapPredictionBoundary(unittest.TestCase):
    def test_predict_at_threshold_boundary_s(self):
        # Recall threshold is 0.65.
        # Fact with 20 stemmed words, passage containing 13 matching words -> 13/20 = 0.65 exactly
        checker = WordRecallThresholdFactChecker(threshold=0.65, scorer="recall")
        fact = "word01 word02 word03 word04 word05 word06 word07 word08 word09 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20"
        # 13 words match
        passage_text = "word01 word02 word03 word04 word05 word06 word07 word08 word09 word10 word11 word12 word13"
        passages = [{"title": "Wiki", "text": passage_text}]
        
        prediction = checker.predict(fact, passages)
        self.assertEqual(prediction, "S")

    def test_predict_just_below_threshold_ns(self):
        # 12 of 20 words match -> 12/20 = 0.60 < 0.65 -> "NS"
        checker = WordRecallThresholdFactChecker(threshold=0.65, scorer="recall")
        fact = "word01 word02 word03 word04 word05 word06 word07 word08 word09 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20"
        passage_text = "word01 word02 word03 word04 word05 word06 word07 word08 word09 word10 word11 word12"
        passages = [{"title": "Wiki", "text": passage_text}]
        
        prediction = checker.predict(fact, passages)
        self.assertEqual(prediction, "NS")


if __name__ == "__main__":
    unittest.main()
