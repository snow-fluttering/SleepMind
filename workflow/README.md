# Local EEG Workflow

Run each public action separately. Transfer returns an exact `next_action.command`
only after success. Stage requires private trained assets, and verify checks saved
results; neither action trains or selects a replacement model.

Set `SLEEPMIND_ASSET_ROOT` to an external directory containing the matched `runs/`
and `reports/` tree. It must supply the five member checkpoints, their configurations
and fitted normalization, the `ens_CE` bundle, learned transition and frozen protocol.
The source repository includes only expected file hashes. `health_check.py --assets`
checks all 20 fixed trained files; ordinary health checks require no model or dataset.

Inputs are finite, single-channel numeric text with a same-directory
`source_manifest.json` declaring exact samples, bytes, SHA-256, raw unfiltered state,
100 Hz, V, and no labels. Transfer validates this contract, loopback chunk order and
token identity, rejects source changes, and records finalized received-file hashes.

Stage invokes the executing repository's `src/predict.py` with external absolute
run and protocol paths. It checks trained assets and received contents before and
after inference, and records executing source hashes separately from model hashes.
The frozen ensemble is `ens_CE` with learned Viterbi strength 0.25.

Verify rechecks file and source consistency, shapes and numeric values, recomputes
argmax, confidence and Viterbi separately within each scorable run, and checks TSV
labels, class durations, coverage and Unknown tails. A record with no complete
30-second epoch remains entirely Unknown. Failed verification removes a stale summary.

These checks establish research-output consistency. Offline filtering and
bidirectional inference use future context. File hashes are not signatures, and
the old frozen protocol and transferred ensemble confidence remain unvalidated
for calibration, real-time devices or clinical use.
