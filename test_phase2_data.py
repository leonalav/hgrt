"""Tests for Phase 2: long-form prose data and target budgets."""
import unittest

from phase2_data import (
    CHAT_TARGET_BYTES,
    PROSE_TARGET_BYTES,
    QA_TARGET_BYTES,
    long_form_paragraphs,
    split_sentences,
    split_wiki_markup,
    target_budget_for,
)


DOC = (
    "The harbour was quiet that morning. Fishing boats sat idle along the pier, "
    "their nets folded on the decks. A gull landed on the capstan and watched "
    "the harbourmaster without blinking. He checked his ledger twice before "
    "shouting for the tide. By noon the market was crowded with voices. "
)


class SentenceSplitTests(unittest.TestCase):
    def test_splits_on_terminators(self):
        sentences = split_sentences(DOC)
        self.assertEqual(len(sentences), 5)
        self.assertTrue(sentences[0].startswith('The harbour'))
        self.assertTrue(sentences[-1].startswith('By noon'))

    def test_abbreviations_are_not_boundaries(self):
        text = "Mr. Smith arrived. He was late. "
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].startswith('Mr. Smith'))

    def test_initials_are_not_boundaries(self):
        text = "J. R. Smith wrote it. Then he stopped. "
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)

    def test_empty_text(self):
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences('   '), [])

    def test_text_without_terminator_is_one_sentence(self):
        self.assertEqual(len(split_sentences('no terminator here')), 1)


class LongFormParagraphTests(unittest.TestCase):
    def test_yields_prompt_target_pairs(self):
        pairs = list(long_form_paragraphs(DOC))
        self.assertGreater(len(pairs), 0)
        for prompt, target in pairs:
            self.assertTrue(prompt)
            self.assertTrue(target)

    def test_target_is_later_than_prompt(self):
        for prompt, target in long_form_paragraphs(DOC):
            self.assertLess(len(prompt), len(prompt) + len(target))
            self.assertTrue(DOC.startswith(prompt[:20]))

    def test_target_respects_the_byte_budget(self):
        for prompt, target in long_form_paragraphs(DOC, max_bytes=120):
            self.assertLessEqual(len(target.encode()), 120)

    def test_short_document_yields_nothing(self):
        self.assertEqual(list(long_form_paragraphs("Only one sentence here.")), [])


class BudgetTests(unittest.TestCase):
    def test_qa_keeps_a_short_budget(self):
        self.assertEqual(target_budget_for('qa'), QA_TARGET_BYTES)
        self.assertEqual(target_budget_for('qa_excerpt'), QA_TARGET_BYTES)
        self.assertEqual(target_budget_for('qa_extractive'), QA_TARGET_BYTES)

    def test_chat_keeps_a_short_budget(self):
        self.assertEqual(target_budget_for('chat'), CHAT_TARGET_BYTES)

    def test_prose_gets_a_long_budget(self):
        self.assertEqual(target_budget_for('prose'), PROSE_TARGET_BYTES)
        self.assertEqual(target_budget_for('prose_long'), PROSE_TARGET_BYTES)

    def test_prose_budget_exceeds_the_old_160_byte_cap(self):
        """Diagnosis 4: the old 160-byte cap forced short padded targets."""
        self.assertGreater(PROSE_TARGET_BYTES, 160)


class MarkupTests(unittest.TestCase):
    def test_removes_wikitext_tokens(self):
        self.assertEqual(split_wiki_markup('a @-@ b @,@ c @.@ d'), 'a - b , c . d')

    def test_leaves_ordinary_text_alone(self):
        self.assertEqual(split_wiki_markup('ordinary text'), 'ordinary text')


if __name__ == '__main__':
    unittest.main()
