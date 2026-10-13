"""Participant grouping, manifest validation, and CV using generated record metadata."""
import copy
import json
import sys

import pytest

from data_preprocessing import participant_id
import splits


def _records():
    rows = []
    for index in range(12):
        for night in (1, 2):
            rows.append({'record_id': f'ST8{index:02d}{night}J0', 'epoch_counts': [10 + index, 3, 20, 8, 9]})
    for index in range(4):
        rows.append({'record_id': f'SC90{index}1E0', 'epoch_counts': [50, 2, 10, 5, 6]})
    return rows


def test_nights_stay_together_and_partitions_are_disjoint_and_seeded():
    kwargs = dict(forced_holdout=['ST800'], holdout_extra={'SC': 1}, validation={'SC': 1, 'ST': 2}, trials=30)
    first = splits.make_split(_records(), 7, **kwargs)
    again = splits.make_split(_records(), 7, **kwargs)
    assert first == again and first['manifest_sha256'] == splits.manifest_hash(first)
    assert 'ST800' in first['partitions']['holdout']['participants']
    owner = {}
    for name in splits.PARTITIONS:
        for record in first['partitions'][name]['records']:
            owner.setdefault(participant_id(record), set()).add(name)
    assert all(len(v) == 1 for v in owner.values())
    assert len(first['partitions']['validation']['cohort_participants']['SC']) == 1
    different = splits.make_split(_records(), 8, **kwargs)
    assert different['partitions'] != first['partitions']


def test_verification_rejects_leakage_and_tampering():
    manifest = splits.make_split(_records(), 3, validation={'ST': 2}, holdout_extra={'ST': 2}, trials=10)
    leaked = copy.deepcopy(manifest)
    moved = leaked['partitions']['validation']['participants'][0]
    leaked['partitions']['holdout']['participants'].append(moved)
    with pytest.raises(ValueError, match='appears in'):
        splits.verify_manifest(leaked)
    split_night = copy.deepcopy(manifest)
    record = split_night['partitions']['train']['records'].pop()
    split_night['partitions']['holdout']['records'].append(record)
    with pytest.raises(ValueError, match='split from its participant'):
        splits.verify_manifest(split_night)
    tampered = copy.deepcopy(manifest)
    tampered['seed'] += 1
    with pytest.raises(ValueError, match='hash mismatch'):
        splits.verify_manifest(tampered)
    with pytest.raises(ValueError, match='not in corpus'):
        splits.make_split(_records(), 1, forced_holdout=['ST999'])


def test_saved_manifest_keeps_forced_holdout_participants_and_repeat_nights(tmp_path):
    forced = ['ST801', 'ST803', 'ST807']
    generated = splits.make_split(_records(), 11, forced_holdout=forced,
        holdout_extra={'SC': 1}, validation={'SC': 1, 'ST': 2}, trials=30)
    path = tmp_path / 'split.json'
    path.write_text(json.dumps(generated), encoding='ascii')
    manifest = splits.load_manifest(path)
    assert manifest == generated
    parts = manifest['partitions']
    assert sum(len(p['participants']) for p in parts.values()) == 16
    assert sum(len(p['records']) for p in parts.values()) == 28
    assert set(forced) <= set(parts['holdout']['participants'])
    siblings = {r for p in forced for r in manifest['participants'][p]['records']}
    assert siblings <= set(parts['holdout']['records'])
    for name in splits.PARTITIONS:
        assert all(parts[name]['stage_counts'][s] > 0 for s in splits.STAGES)
        assert parts[name]['cohort_participants']['SC'] and parts[name]['cohort_participants']['ST']


def test_cv_folds_validate_every_development_participant_once_and_keep_the_holdout():
    holdout = ['ST800', 'SC900']
    folds = splits.make_cv_folds(_records(), holdout, 4, seed=5, trials=40)
    assert len(folds) == 4
    development = {p for p in folds[0]['participants'] if p not in holdout}
    validated = [p for f in folds for p in f['partitions']['validation']['participants']]
    assert sorted(validated) == sorted(development)
    for fold in folds:
        assert fold['partitions']['holdout']['participants'] == sorted(holdout)
        assert set(fold['partitions']['train']['participants']).isdisjoint(fold['partitions']['validation']['participants'])
        splits.verify_manifest(fold)
    with pytest.raises(ValueError):
        splits.make_cv_folds(_records(), holdout, 1, seed=5)


def test_saved_cv_folds_keep_holdout_and_validate_each_development_participant_once(tmp_path):
    fixed = splits.make_split(_records(), 23, forced_holdout=['ST800'],
        holdout_extra={'SC': 1}, validation={'SC': 1, 'ST': 2}, trials=30)
    generated = splits.make_cv_folds(_records(), fixed['partitions']['holdout']['participants'],
        5, seed=29, trials=40)
    seen = []
    for index, original in enumerate(generated):
        path = tmp_path / f'fold_{index}.json'
        path.write_text(json.dumps(original), encoding='ascii')
        fold = splits.load_manifest(path)
        assert fold == original
        assert fold['partitions']['holdout'] == fixed['partitions']['holdout']
        seen += fold['partitions']['validation']['participants']
    development = fixed['partitions']['train']['participants'] + fixed['partitions']['validation']['participants']
    assert sorted(seen) == sorted(development)


@pytest.mark.parametrize('options', [[], ['--forced_holdout', 'ST800']])
def test_split_cli_requires_explicit_partitions_before_reading_inventory(monkeypatch, options):
    def unexpected_inventory_read(*args):
        raise AssertionError('Missing partition intent must be rejected before reading inventory')

    monkeypatch.setattr(splits, 'records_from_inventory', unexpected_inventory_read)
    monkeypatch.setattr(sys, 'argv', ['splits.py', *options])
    with pytest.raises(SystemExit) as exc:
        splits.main()
    assert exc.value.code == 2


def test_split_cli_creates_only_explicit_synthetic_partitions(tmp_path, monkeypatch):
    inventory = tmp_path / 'inventory.json'
    rows = [{'mode': 'raw', 'split': 'train', 'record_id': row['record_id'],
             'full_valid_epoch_counts': dict(zip(splits.STAGES, row['epoch_counts']))}
            for row in _records()]
    inventory.write_text(json.dumps({'records': rows}), encoding='ascii')
    output = tmp_path / 'split.json'
    monkeypatch.setattr(sys, 'argv', ['splits.py', '--inventory', str(inventory),
        '--output', str(output), '--forced_holdout', 'ST800', '--holdout_extra', 'SC=1',
        '--validation', 'SC=1,ST=2'])
    splits.main()
    manifest = splits.load_manifest(output)
    assert manifest['seed'] == 42
    assert 'ST800' in manifest['partitions']['holdout']['participants']
    assert all(manifest['partitions'][name]['participants'] for name in splits.PARTITIONS)
