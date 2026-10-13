"""Regression checks for restoration, alignment, continuity, and cache validity."""
import gc
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_preprocessing as prep
from dataset import run_table, window_starts


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def part(self, number, values, labels=None, filtered=True, sid="SC9001E0"):
        folder = self.root / f"part_{number}"
        folder.mkdir(exist_ok=True)
        suffix = "_filtered_eeg_data" if filtered else ""
        eeg = folder / f"{sid}_EEGFpz_Cz_Part_{number}_of_5{suffix}.txt"
        np.savetxt(eeg, np.asarray(values), fmt="%.10g")
        label_path = None
        if labels is not None:
            label_path = folder / f"{sid}_Hypnogram_Data_Part_{number}_of_5.txt"
            label_path.write_text("onset_sec end_sec duration_sec description\n" + "\n".join(
                f"{a} {b} {b-a} {desc}" for a, b, desc in labels), encoding="ascii")
        return str(eeg), str(label_path) if label_path else None

    def test_recursive_filtered_scanner_and_duplicate_rejection(self):
        self.part(1, np.arange(35), [(0, 35, "Sleep stage W")])
        self.assertEqual(list(prep.scan_dataset(self.root)), ["SC9001E0"])
        duplicate = self.root / "duplicate"
        duplicate.mkdir()
        source = Path(prep.scan_dataset(self.root)["SC9001E0"][1][0])
        (duplicate / source.name).write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate EEG"):
            prep.scan_dataset(self.root)

    def test_corrupt_samples_are_not_silently_removed(self):
        bad = self.root / "bad.txt"
        for content in ("1\ncorrupt\n2\n", "1\nnan\n", "1\ninf\n", "1 2\n3 4\n"):
            bad.write_text(content, encoding="ascii")
            with self.assertRaises(ValueError):
                prep.load_eeg_txt(bad)

    def test_global_labels_and_mid_epoch_part_cuts_stitch_exactly(self):
        parts = {1: self.part(1, np.arange(35), [(0, 35, "Sleep stage W")]),
                 2: self.part(2, np.arange(35, 90), [(35, 60, "Sleep stage W"),
                                                    (60, 90, "Sleep stage 2")])}
        with patch.object(prep, "apply_filter", side_effect=AssertionError("double filtering")):
            record = prep.load_record(parts, fs=1)
            X, y = prep.preprocess_subject(parts, fs=1, epoch_sec=30)
        np.testing.assert_array_equal(record["eeg"], np.arange(90))
        np.testing.assert_array_equal(X, np.arange(90).reshape(3, 30))
        np.testing.assert_array_equal(y, [0, 0, 2])
        self.assertEqual(record["part_metadata"][1]["start_sample"], 35)

    def test_raw_filter_runs_once_after_joining_consecutive_parts(self):
        parts = {1: self.part(1, np.arange(35), filtered=False),
                 2: self.part(2, np.arange(35, 90), filtered=False)}
        with patch.object(prep, "apply_filter", side_effect=lambda x, *a: x + 100) as filtering:
            record = prep.load_record(parts, fs=1, input_mode="raw")
        self.assertEqual(filtering.call_count, 1)
        np.testing.assert_array_equal(record["eeg"], np.arange(90) + 100)

    def test_missing_signal_part_creates_a_timestamped_hole(self):
        parts = {1: self.part(1, np.arange(30), [(0, 30, "Sleep stage W")]),
                 3: self.part(3, np.arange(60, 90), [(60, 90, "Sleep stage 2")])}
        with self.assertRaisesRegex(ValueError, "explicit part_start_samples"):
            prep.load_record(parts, fs=1)
        record = prep.load_record(parts, fs=1, part_start_samples={3: 60})
        self.assertEqual(len(record["eeg"]), 90)
        self.assertTrue(np.isnan(record["eeg"][30:60]).all())
        self.assertEqual(record["blocks"][1]["start_sample"], 60)
        X, y, sid = prep.build_training_set(self.root, fs=1, epoch_sec=30, verbose=False,
                                          record_part_start_samples={"SC9001E0": {3: 60}})
        np.testing.assert_array_equal(y, [0, 2])
        self.assertNotEqual(sid[0], sid[1])
        starts, lengths = run_table(sid)
        self.assertEqual(len(window_starts(starts, lengths, 2, stride=1)), 0)

    def test_invalid_labels_reset_temporal_context(self):
        self.part(1, np.arange(150), [(0, 60, "Sleep stage 2"), (60, 90, "Sleep stage ?"),
                                     (90, 150, "Sleep stage 2")])
        X, y, sid = prep.build_training_set(self.root, fs=1, epoch_sec=30, verbose=False)
        starts, lengths = run_table(sid)
        windows = window_starts(starts, lengths, 2, stride=1)
        np.testing.assert_array_equal(windows, [0, 2])  # no window spans the unknown interval
        np.testing.assert_array_equal(X[windows[1]:windows[1] + 2], np.arange(90, 150).reshape(2, 30))
        np.testing.assert_array_equal(y[windows[1]:windows[1] + 2], [2, 2])

    def test_blank_test_labels_remain_unknown(self):
        _, path = self.part(1, np.arange(60), [(0, 30, "Sleep stage W"), (30, 60, "")])
        parsed = prep.parse_label_file(path)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(prep.map_description_to_label(parsed[1][3]), -1)

    def test_cache_changes_when_source_or_processing_changes(self):
        _, label = self.part(1, np.arange(60), [(0, 60, "Sleep stage W")])
        cache = self.root / "cache.npz"
        _, y, _ = prep.build_training_set(self.root, fs=1, epoch_sec=30, cache_path=cache, verbose=False)
        np.testing.assert_array_equal(y, [0, 0])
        Path(label).write_text("onset_sec end_sec duration_sec description\n0 60 60 Sleep stage 2\n", encoding="ascii")
        X, y, _ = prep.build_training_set(self.root, fs=1, epoch_sec=30, cache_path=cache, verbose=False)
        np.testing.assert_array_equal(y, [2, 2])
        normalized, _, _ = prep.build_training_set(self.root, fs=1, epoch_sec=30,
                                                  epoch_normalize="zscore", cache_path=cache, verbose=False)
        self.assertFalse(np.array_equal(X, normalized))
        np.testing.assert_allclose(normalized.mean(axis=1), 0, atol=1e-6)

    def test_record_cache_is_content_addressed_and_filters_once(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(scale=1e-5, size=(2, 6000))
        parts = {1: self.part(1, samples[0], [(0, 60, "Sleep stage 2")], filtered=False),
                 2: self.part(2, samples[1], [(60, 120, "Sleep stage 2")], filtered=False)}
        cache = self.root / "records"
        direct = prep.load_record(parts, fs=100, input_mode="raw")
        first = prep.load_record_cached("SC9001E0", parts, cache, fs=100, input_mode="raw")
        np.testing.assert_allclose(first["eeg"], direct["eeg"])  # parts joined, then filtered once
        with patch.object(prep, "load_record", side_effect=AssertionError("cache should serve this")):
            again = prep.load_record_cached("SC9001E0", parts, cache, fs=100, input_mode="raw")
        np.testing.assert_array_equal(again["eeg"], first["eeg"])
        self.assertEqual(again["blocks"], first["blocks"])
        pre = prep.load_record_cached("SC9001E0", parts, cache, fs=100, input_mode="preprocessed")
        self.assertFalse(np.allclose(pre["eeg"], first["eeg"]))  # processing mode changes the key
        bytes_before = Path(parts[1][0]).read_bytes()
        Path(parts[1][0]).write_bytes(bytes_before.replace(b"e-0", b"e-1", 1))
        changed = prep.load_record_cached("SC9001E0", parts, cache, fs=100, input_mode="raw")
        self.assertFalse(np.array_equal(changed["eeg"], first["eeg"]))  # edited content invalidates
        del first, again, pre, changed  # release memory maps so Windows can remove the temporary folder
        gc.collect()

    def test_two_nights_group_into_one_participant(self):
        self.assertEqual(prep.participant_id("SC9001E0::segment_2"), "SC900")
        self.assertEqual(prep.participant_id("SC9002E0::segment_0"), "SC900")
        self.assertEqual(prep.participant_id("ST8191J0"), "ST819")
        self.assertEqual(prep.participant_id("SC01::segment_0"), "SC01")

    def test_same_size_same_timestamp_mutation_invalidates_cache(self):
        _, label = self.part(1, np.arange(60), [(0, 60, "Sleep stage 1")])
        cache = self.root / "cache.npz"
        prep.build_training_set(self.root, fs=1, epoch_sec=30, cache_path=cache, verbose=False)
        before = Path(label).stat()
        Path(label).write_bytes(Path(label).read_bytes().replace(b"Sleep stage 1", b"Sleep stage 2"))
        os.utime(label, ns=(before.st_atime_ns, before.st_mtime_ns))
        _, y, _ = prep.build_training_set(self.root, fs=1, epoch_sec=30, cache_path=cache, verbose=False)
        np.testing.assert_array_equal(y, [2, 2])


if __name__ == "__main__":
    unittest.main()
