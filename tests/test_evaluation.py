"""Scientific evaluation checks for timestamps, abstentions, and partial labels."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils import evaluate, evaluate_arrays


class EvaluationTests(unittest.TestCase):
    def test_unknown_is_excluded_and_wake_only_is_not_sleep_accuracy(self):
        result = evaluate_arrays([-1, 0, 0, -1], [4, 0, 0, 2])
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['macro_f1_5class'], .2)
        self.assertEqual(result['scored_epochs'], 2)
        self.assertFalse(result['sleep_stage_evaluation_available'])
        self.assertIsNone(result['sleep_only_macro_f1_excluding_reference_wake'])

    def test_rejected_predictions_are_visible_and_cannot_improve_coverage(self):
        result = evaluate_arrays([0, 1, 2, 4], [0, -1, 2, 4], scorable=[1, 1, 1, 0])
        self.assertEqual(result['scored_epochs'], 2)
        self.assertEqual(result['rejected_predicted_epochs'], 1)
        self.assertEqual(result['known_but_unscorable_epochs'], 1)
        self.assertAlmostEqual(result['prediction_coverage_on_eligible_reference'], 2 / 3)
        self.assertAlmostEqual(result['correct_fraction_of_eligible_reference'], 2 / 3)
        self.assertEqual(result['reference_stage_counts']['REM'], 0)

    def test_length_mismatch_and_invalid_labels_fail(self):
        with self.assertRaises(ValueError):
            evaluate_arrays([0, 1], [0])
        with self.assertRaises(ValueError):
            evaluate_arrays([0, 5], [0, 1])

    def test_npz_compares_decoding_and_rejects_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'example_epochs.npz'
            np.savez(file, onset_sec=[0, 30, 60], ground_truth=[0, 1, -1],
                     argmax=[0, 2, 4], decoded=[0, 1, 4])
            result = evaluate(directory)
            self.assertEqual(result['argmax_decoded_disagreements'], 1)
            self.assertEqual(result['argmax']['accuracy'], .5)
            self.assertEqual(result['decoded']['accuracy'], 1.0)
            np.savez(file, onset_sec=[0, 0], ground_truth=[0, 1], argmax=[0, 1], decoded=[0, 1])
            with self.assertRaisesRegex(ValueError, 'duplicates'):
                evaluate(directory)

    def test_segment_alignment_uses_absolute_time_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, references = root / 'pred', root / 'gt'
            predictions.mkdir()
            references.mkdir()
            header = 'onset_sec\tend_sec\tduration_sec\tdescription\n'
            (predictions / 'example_hypnogram.tsv').write_text(
                header + '0\t30\t30\tSleep stage 1\n30\t60\t30\tSleep stage 2\n', encoding='utf-8')
            (references / 'example_Hypnogram_Data_Part_2_of_5.txt').write_text(
                header + '30\t60\t30\tSleep stage 1\n60\t90\t30\tSleep stage 2\n', encoding='utf-8')
            result = evaluate(predictions, references)['predictions']
            self.assertEqual(result['known_reference_epochs'], 2)
            self.assertEqual(result['accuracy'], 0.0)
            self.assertEqual(result['rejected_predicted_epochs'], 1)
            self.assertEqual(result['total_epoch_positions'], 3)

    def test_identical_label_across_part_boundary_covers_whole_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt = root / 'pred', root / 'gt'
            pred.mkdir()
            gt.mkdir()
            header = 'onset_sec\tend_sec\tduration_sec\tdescription\n'
            (pred / 'example_hypnogram.tsv').write_text(header + '0\t60\t60\tSleep stage 2\n', encoding='utf-8')
            (gt / 'example_Hypnogram_Data_Part_1_of_5.txt').write_text(header + '0\t42\t42\tSleep stage 2\n', encoding='utf-8')
            (gt / 'example_Hypnogram_Data_Part_2_of_5.txt').write_text(header + '42\t60\t18\tSleep stage 2\n', encoding='utf-8')
            result = evaluate(pred, gt)['predictions']
            self.assertEqual(result['scored_epochs'], 2)
            self.assertEqual(result['accuracy'], 1.0)

    def test_empty_path_and_no_matching_files_fail(self):
        with self.assertRaises(ValueError):
            evaluate('')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                evaluate(directory)

    def test_missing_whole_record_prediction_reduces_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred, gt = root / 'pred', root / 'gt'
            pred.mkdir()
            gt.mkdir()
            header = 'onset_sec\tend_sec\tduration_sec\tdescription\n'
            (pred / 'present_hypnogram.tsv').write_text(header + '0\t30\t30\tWake\n', encoding='utf-8')
            (gt / 'present_hypnogram.tsv').write_text(header + '0\t30\t30\tWake\n', encoding='utf-8')
            (gt / 'missing_hypnogram.tsv').write_text(header + '0\t30\t30\tN2\n', encoding='utf-8')
            result = evaluate(pred, gt)
            self.assertEqual(result['reference_recordings_without_prediction'], ['missing'])
            self.assertEqual(result['predictions']['known_reference_epochs'], 2)
            self.assertEqual(result['predictions']['rejected_predicted_epochs'], 1)
            self.assertEqual(result['predictions']['prediction_coverage_on_eligible_reference'], .5)


if __name__ == '__main__':
    unittest.main()
