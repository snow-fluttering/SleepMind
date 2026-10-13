"""Research sleep summaries with explicit unscorable coverage and no diagnosis."""
import html
import json
from pathlib import Path
import numpy as np

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def validate_segments(segments, total_seconds=None):
    if total_seconds is not None and (not np.isfinite(total_seconds) or total_seconds <= 0):
        raise ValueError('Recording duration must be finite and positive')
    previous_end = 0.0
    for index, segment in enumerate(segments):
        start, end, label = segment['onset_sec'], segment['end_sec'], segment['label']
        if not np.isfinite([start, end]).all() or end <= start or start < previous_end - 1e-6:
            raise ValueError(f'Invalid or overlapping segment at index {index}')
        if label not in (-1, 0, 1, 2, 3, 4):
            raise ValueError('Invalid stage label')
        if total_seconds is not None and end > total_seconds + 1e-6:
            raise ValueError('Segment extends beyond the recording')
        previous_end = end
    if total_seconds is not None:
        if not segments or abs(segments[0]['onset_sec']) > 1e-6 or abs(previous_end - total_seconds) > 1e-6:
            raise ValueError('Segments must cover the complete recording timeline')
        for left, right in zip(segments, segments[1:]):
            if abs(left['end_sec'] - right['onset_sec']) > 1e-6:
                raise ValueError('Missing intervals must be represented explicitly as unscorable')
    return True


def summarize(segments, metadata):
    total = float(metadata['total_seconds'])
    validate_segments(segments, total)
    seconds = {name: 0.0 for name in STAGES}
    unknown = 0.0
    for segment in segments:
        duration = segment['end_sec'] - segment['onset_sec']
        if segment['label'] == -1:
            unknown += duration
        else:
            seconds[STAGES[segment['label']]] += duration
    sleep = sum(seconds[stage] for stage in STAGES[1:])
    scored = total - unknown
    first_sleep = next((s['onset_sec'] for s in segments if s['label'] in (1, 2, 3, 4)), None)
    limitations = [
        'Flat signal frames and incomplete epochs are unscorable, and missing intervals reset temporal context.',
        'Offline filtering and bidirectional inference use future context; streaming BCI validation is pending.',
        'Short continuous runs use only observed context; their accuracy requires separate validation.',
    ]
    if metadata.get('imported_legacy_checkpoint'):
        limitations.append('Participant isolation in the imported checkpoint training and validation is unverified.')
    if metadata.get('known_annotation_classes') == [0]:
        limitations.append('Available annotations contain wake only; sleep-stage accuracy cannot be measured from them.')
    if metadata.get('transition_is_heuristic') and metadata.get('decoder') == 'viterbi':
        limitations.append('The default Viterbi transition prior is a configurable heuristic requiring validation.')
    if metadata.get('checkpoint_input_mode', 'unknown') == 'unknown':
        limitations.append('Legacy checkpoint preprocessing provenance is unknown; current input processing is explicit.')
    elif metadata.get('checkpoint_input_mode') != metadata.get('input_mode'):
        limitations.append('Input processing mode differs from checkpoint training; this is a distribution-shift experiment.')
    confidence = metadata.get('confidence')
    if confidence and confidence.get('calibration_transferred_to_ensemble'):
        limitations.append('Temperature and confidence threshold were transferred from fold-model predictions; '
                           'calibration of the deployed ensemble remains unvalidated.')
    elif confidence and not confidence.get('calibrated'):
        limitations.append('Reported confidence is an uncalibrated softmax score.')
    if metadata.get('protocol') and not metadata.get('protocol_inference_contract_bound', False):
        limitations.append('The saved protocol binds checkpoint weights only; its preprocessing and inference settings '
                           'were not bound by the original protocol artifact.')
    return {
        'schema_version': 1,
        'purpose': 'Research sleep staging; not a medical diagnosis',
        'metadata': metadata,
        'recording_seconds': total,
        'scorable_seconds': scored,
        'unscorable_seconds': unknown,
        'scorable_fraction': scored / total if total else None,
        'stage_minutes': {key: value / 60 for key, value in seconds.items()},
        'estimated_sleep_minutes': sleep / 60,
        'sleep_fraction_of_scorable_time': sleep / scored if scored else None,
        'clinical_sleep_efficiency': None,
        'clinical_sleep_efficiency_reason': 'Time in bed is not provided',
        'first_observed_sleep_seconds': first_sleep,
        'confidence': confidence,
        'limitations': limitations,
    }


def write_report(segments, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sid = metadata['record_id']
    report = summarize(segments, metadata)
    (output_dir / f'{sid}_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    rows = ''.join(f'<tr><td>{stage}</td><td>{minutes:.1f}</td></tr>' for stage, minutes in report['stage_minutes'].items())
    confidence = report.get('confidence')
    confidence_html = ''
    if confidence and confidence.get('mean_top_probability') is not None:
        flagged = confidence['low_confidence_epochs']
        confidence_kind = ('transferred temperature; ensemble validation pending'
                           if confidence.get('calibration_transferred_to_ensemble') else
                           'temperature-scaled on validation data' if confidence['calibrated'] else 'uncalibrated')
        confidence_html = (f"<p>Mean top-class probability: {confidence['mean_top_probability']:.3f} "
                           f"({confidence_kind})."
                           + (f" Low-confidence epochs (below {confidence['low_confidence_threshold']}): {flagged} of {confidence['scorable_epochs']}." if flagged is not None else '')
                           + f" {html.escape(confidence['note'])}</p>")
    limitations = ''.join(f'<li>{html.escape(item)}</li>' for item in report['limitations'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(sid)} Sleep Report</title>
<style>body{{font:16px system-ui;max-width:960px;margin:40px auto;color:#17324d;padding:0 24px}}table{{border-collapse:collapse}}td,th{{padding:10px 22px;border-bottom:1px solid #ddd}}img{{width:100%}}small{{color:#52687d}}</style>
<h1>{html.escape(sid)} Sleep Report</h1><p>{html.escape(report['purpose'])}</p>
<p>Recording: {total_minutes(report):.1f} minutes. Scorable: {report['scorable_fraction'] or 0:.1%}.
Estimated sleep: {report['estimated_sleep_minutes']:.1f} minutes. Unscorable: {report['unscorable_seconds']/60:.1f} minutes.</p>
{confidence_html}<img src="{html.escape(sid)}_hypnogram.png" alt="Sleep stage timeline with gaps shown as unscorable">
<table><tr><th>Stage</th><th>Minutes</th></tr>{rows}</table><h2>Interpretation limits</h2><ul>{limitations}</ul>
<small>Reproducible settings, checkpoint hash, data paths and timing are stored in the adjacent JSON file.</small></html>'''
    (output_dir / f'{sid}_report.html').write_text(page, encoding='utf-8')
    plot_segments(segments, output_dir / f'{sid}_hypnogram.png', sid)
    return report


def total_minutes(report):
    return report['recording_seconds'] / 60


def plot_segments(segments, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    remap = {-1: -1, 0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    fig, ax = plt.subplots(figsize=(12, 3))
    for segment in segments:
        start, end = segment['onset_sec'] / 3600, segment['end_sec'] / 3600
        ax.plot([start, end], [remap[segment['label']]] * 2, color='#1e7293' if segment['label'] >= 0 else '#999', linewidth=1.2)
    ax.set_yticks([-1, 0, 1, 2, 3, 4], ['Unscorable', 'N3', 'N2', 'N1', 'REM', 'W'])
    ax.set_xlabel('Time from recording start (hours)')
    ax.set_title(title)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
