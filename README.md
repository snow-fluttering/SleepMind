# SleepMind

SleepMind V3.1 is a reproducible, offline research pipeline for single-channel EEG sleep analysis. It reads recorded signals, filters raw input, performs temporal deep-learning inference, applies Viterbi decoding, preserves missing intervals, validates results, and generates sleep research reports.

## Repository contents

- `src/`: readers, preprocessing, model, training, inference, decoding, evaluation, and reports.
- `workflow/`: separate transfer, stage, and verify actions with structured results.
- `bci_replay/`: local TCP replay and ordered signal reception.
- `tools/`: grouped cross-validation, ensemble construction, evaluation, and inference benchmarks.
- `tests/`: self-contained synthetic regression tests.

Datasets, recordings, trained weights, fitted normalization, learned priors, frozen protocols, experiment records, and generated reports are supplied separately. This repository contains source only. Real sleep staging requires a complete compatible private model bundle; missing assets never trigger random-model inference.

## Installation

Use Python 3.12. For the tested CPU setup:

```text
python setup_env.py
python run_action.py --check
```

The installer creates a local virtual environment, installs CPU PyTorch 2.12.1 from the official wheel index and the pinned numerical dependencies, and checks source files and imports. It does not download datasets or trained models. The launcher selects `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` on other systems.

For a manually managed environment:

```text
python -m venv .venv
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

Run the last two commands with the virtual environment's Python after activation. Different hardware and dependency builds can change floating-point results.

## Tests

Activate the `.venv` environment first, then install the development dependencies and run the tests:

```text
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests generate their own temporary signals, annotations, and manifests. They cover preprocessing, filtering state, absolute alignment, gaps, partial epochs, participant isolation, training-only statistics, decoding, numerical validity, shared-encoder equivalence, checkpoint loading, and output integrity. They do not measure sleep-stage accuracy on a real cohort.

## Input contract

The text adapter accepts one numeric EEG value per sample. The current inference contract is one channel at 100 Hz, with 30-second epochs. Declare units explicitly as `V` or `uV` when using the core text reader.

Raw continuous blocks receive one fourth-order 0.5-30 Hz Butterworth SOS forward-backward filter. Preprocessed inputs bypass filtering. Preserve original sample offsets and timestamps. Missing samples, flat signals, unknown references, and incomplete epochs remain explicit; they do not become Wake or bridge disconnected sequences.

Other acquisition devices and sampling rates require a validated reader and resampling contract. The TCP component replays a recorded file locally; it does not establish wearable-device or real-time performance.

## External model assets and the three actions

Set `SLEEPMIND_ASSET_ROOT` to a private directory containing the compatible `runs/` and `reports/` model assets. The integrated workflow preserves the selected five-member `ens_CE` mean-logit ensemble and its frozen learned Viterbi strength of `0.25`. Twenty trained asset hashes are checked independently from the executing repository source hashes.

Set the environment variable in the same terminal that runs the actions. For PowerShell:

```powershell
$env:SLEEPMIND_ASSET_ROOT = 'D:/private/model-assets'
```

For a POSIX shell:

```sh
export SLEEPMIND_ASSET_ROOT=/path/to/private/model-assets
```

The optional `--check --assets PATH` validates that path for the health check only; it does not set the environment variable for subsequent actions.

```text
python run_action.py --check --assets /path/to/private/model-assets
python run_action.py transfer --input /path/to/recording.txt
```

The transfer entry requires a matching `source_manifest.json` beside the input. This entry accepts only declared raw, unfiltered, single-channel 100 Hz data in volts. Sampling rate and units are declarations, not automatic detection. Include all fields below, replacing the illustrative counts and hash with the exact input file's values:

```json
{
  "sample_count": 3000,
  "extracted_bytes": 27000,
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "sampling_rate_contract_hz": 100,
  "amplitude_unit_contract": "V",
  "amplitude_unit_auto_detected": false,
  "processing_state": "raw_unfiltered",
  "labels_read": false
}
```

Do not label a previously filtered file as raw. The zero hash above is a placeholder and will be rejected for a real input. Inspect `python workflow/entry.py transfer --help` for transport options.

After a successful transfer, execute the returned `next_action.command` for stage. After stage succeeds, execute its returned verify command. Keep these as three separate actions, and stop on failure. Transfer verifies received bytes; stage checks the external assets before and after inference; verify recomputes labels and confidence from saved logits, checks timeline coverage, and removes stale summaries on failure.

The core predictor also supports a direct input file and preprocessed mode:

```text
python src/predict.py --run_name /path/to/private/model-assets/runs/ens_CE --source_file /path/to/recording.txt --unit V --input_mode raw --protocol /path/to/private/model-assets/reports/ens_CE/frozen_protocol.json --output_dir reports/new_recording
```

Use `--input_mode preprocessed` for already filtered signals. Apply a frozen protocol only where its input and calibration contracts are valid. `python src/predict.py --help` describes the available options.

## Training and evaluation

Training code is included; training data and historical experiment records are not. New experiments require an explicit participant-disjoint split and start from random initialization. Group all nights and fragments by source participant. Fit normalization, class weights, and transition priors on training participants only; select calibration and decoder settings on development validation data while preserving an untouched final holdout.

The split CLI requires explicit nonempty training, validation, and holdout partitions. Development uses grouped cross-validation, without an inner selection loop; reusing fold validation labels for checkpoint selection makes pooled development estimates optimistic. Do not select a new model or decoder from revealed holdout scores.

Count each original epoch once and use all five fixed classes for primary metrics. Report support, accepted and eligible counts, coverage, abstentions, kappa, participant variation, and paired uncertainty. Paired comparisons require identical participant sets and matched reference support and denominators. A Wake-only reference cannot establish full sleep-stage accuracy.

## Outputs and limitations

Inference writes epoch arrays, timestamped hypnograms, confidence summaries, stage durations, explicit unknown intervals, provenance, and JSON/HTML/PNG research reports. Model-derived priors and confidence thresholds remain exploratory. Coverage and sleep fraction of scored time are not clinical sleep efficiency.

Forward-backward filtering and bidirectional inference use future context. Causal streaming, comprehensive artifact detection, calibration transfer, external-cohort generalization, prospective home-device performance, and clinical validity require additional validation. These outputs are research results.
