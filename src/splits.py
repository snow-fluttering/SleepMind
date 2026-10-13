"""Frozen participant-disjoint train / validation / final-holdout manifests.

Participants (SC/ST prefix groups from ``participant_id``) are assigned as whole
units: every night, part and contiguous run of a participant shares one
partition. Assignment uses only record identifiers and stage counts (labels of
the corpus), never model output. The final holdout is meant to be evaluated once
after all choices are frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data_preprocessing import participant_id

PARTITIONS = ('train', 'validation', 'holdout')
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def cohort_of(participant):
    return str(participant)[:2]


def participant_table(records):
    """Aggregate ``[{record_id, epoch_counts}]`` rows to per-participant totals."""
    table = {}
    for row in records:
        pid = participant_id(row['record_id'])
        entry = table.setdefault(pid, {'participant': pid, 'cohort': cohort_of(pid),
                                       'records': [], 'epoch_counts': [0] * 5})
        if row['record_id'] in entry['records']:
            raise ValueError(f"Duplicate record {row['record_id']}")
        entry['records'].append(row['record_id'])
        counts = [int(v) for v in row['epoch_counts']]
        if len(counts) != 5 or min(counts) < 0:
            raise ValueError('Each record needs five nonnegative stage counts')
        entry['epoch_counts'] = [a + b for a, b in zip(entry['epoch_counts'], counts)]
    for entry in table.values():
        entry['records'] = sorted(entry['records'])
    return table


def _distribution(table, members):
    counts = np.zeros(5)
    for pid in members:
        counts += table[pid]['epoch_counts']
    return counts / counts.sum() if counts.sum() else counts


def _draw(rng, pool, k, table, reference, trials):
    """Seeded candidate search minimizing stage-proportion distance to a reference."""
    if k == 0:
        return []
    if k > len(pool):
        raise ValueError('Requested more participants than available')
    best, best_score = None, np.inf
    for _ in range(trials):
        pick = sorted(rng.choice(pool, size=k, replace=False).tolist())
        score = float(np.abs(_distribution(table, pick) - reference).sum())
        if score < best_score - 1e-12:
            best, best_score = pick, score
    return best


def make_split(records, seed, forced_holdout=(), holdout_extra=None, validation=None, trials=500):
    """Return a manifest dict.

    ``holdout_extra`` / ``validation`` map cohort prefix to a participant count
    drawn (in that order) from participants not already assigned. Candidate
    draws are scored by stage-proportion distance to the cohort-wide
    distribution, so rare stages are represented where feasible.
    """
    table = participant_table(records)
    forced = sorted(set(forced_holdout))
    unknown = [p for p in forced if p not in table]
    if unknown:
        raise ValueError(f'Forced holdout participants not in corpus: {unknown}')
    rng = np.random.default_rng(seed)
    assigned = {'holdout': list(forced), 'validation': []}
    for stage_name, quota in (('holdout', holdout_extra or {}), ('validation', validation or {})):
        for cohort in sorted(quota):
            taken = set(assigned['holdout']) | set(assigned['validation'])
            pool = sorted(p for p, e in table.items() if e['cohort'] == cohort and p not in taken)
            reference = _distribution(table, [p for p, e in table.items() if e['cohort'] == cohort])
            assigned[stage_name].extend(_draw(rng, pool, quota[cohort], table, reference, trials))
    assignment = {p: 'train' for p in table}
    for name in ('validation', 'holdout'):
        for p in assigned[name]:
            assignment[p] = name
    meta = {
        'schema_version': 1, 'seed': seed, 'candidate_trials': trials,
        'forced_holdout_participants': forced,
        'holdout_extra_quota': holdout_extra or {}, 'validation_quota': validation or {},
        'selection_rule': 'Seeded candidate search minimizing L1 distance between partition and '
                          'cohort stage proportions; labels only, no model output.',
    }
    return build_manifest(table, assignment, meta)


def build_manifest(table, assignment, meta):
    """Assemble, verify and hash a manifest from a participant -> partition assignment."""
    manifest = dict(meta)
    manifest['partitions'], manifest['participants'] = {}, {}
    for name in PARTITIONS:
        members = sorted(p for p, a in assignment.items() if a == name)
        recs = sorted(r for p in members for r in table[p]['records'])
        counts = np.sum([table[p]['epoch_counts'] for p in members], axis=0) if members else np.zeros(5, int)
        manifest['partitions'][name] = {
            'participants': members, 'records': recs,
            'cohort_participants': {c: [p for p in members if table[p]['cohort'] == c]
                                    for c in sorted({table[p]['cohort'] for p in table})},
            'labeled_epochs': int(np.sum(counts)),
            'stage_counts': {s: int(c) for s, c in zip(STAGES, counts)},
        }
    for p, entry in sorted(table.items()):
        manifest['participants'][p] = {'partition': assignment[p], 'cohort': entry['cohort'],
                                       'records': entry['records'], 'epoch_counts': entry['epoch_counts']}
    verify_manifest(manifest)
    manifest['manifest_sha256'] = manifest_hash(manifest)
    return manifest


def make_cv_folds(records, holdout_participants, k, seed, trials=3000):
    """Cohort-stratified, stage-balanced participant folds over the non-holdout (development) set.

    Returns ``k`` manifests. In fold i the fold's participants are the validation partition, the remaining
    development participants are training, and the holdout is exactly ``holdout_participants``.
    """
    table = participant_table(records)
    holdout = sorted(set(holdout_participants))
    unknown = [p for p in holdout if p not in table]
    if unknown:
        raise ValueError(f'Holdout participants not in corpus: {unknown}')
    development = sorted(p for p in table if p not in set(holdout))
    if k < 2 or k > len(development):
        raise ValueError('Fold count must lie between 2 and the number of development participants')
    cohorts = sorted({table[p]['cohort'] for p in development})
    overall = _distribution(table, development)
    rng = np.random.default_rng(seed)
    best, best_score = None, np.inf
    for _ in range(trials):
        folds, offset = [[] for _ in range(k)], 0
        for cohort in cohorts:
            members = [p for p in development if table[p]['cohort'] == cohort]
            for index, p in enumerate(rng.permutation(members).tolist()):
                folds[(offset + index) % k].append(p)
            offset += len(members)
        sizes = np.array([sum(sum(table[p]['epoch_counts']) for p in f) for f in folds], float)
        stage = max(float(np.abs(_distribution(table, f) - overall).sum()) for f in folds)
        score = stage + float(sizes.std() / sizes.mean())
        if score < best_score - 1e-12:
            best, best_score = folds, score
    manifests = []
    for index, fold in enumerate(best):
        assignment = {p: 'holdout' for p in holdout}
        assignment.update({p: ('validation' if p in fold else 'train') for p in development})
        meta = {'schema_version': 1, 'kind': 'cross_validation_fold', 'fold': index, 'folds': k, 'seed': seed,
                'candidate_trials': trials, 'fold_balance_score': best_score,
                'selection_rule': 'Seeded search over cohort-stratified assignments minimizing the worst fold '
                                  'stage-proportion L1 distance plus relative fold-size spread; labels only.',
                'holdout_participants': holdout}
        manifests.append(build_manifest(table, assignment, meta))
    covered = sorted(p for m in manifests for p in m['partitions']['validation']['participants'])
    if covered != development:
        raise ValueError('Folds must validate every development participant exactly once')
    return manifests


def manifest_hash(manifest):
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


def verify_manifest(manifest):
    """Raise unless partitions are disjoint at participant and record level and exhaustive."""
    seen_participants, seen_records = {}, {}
    for name in PARTITIONS:
        part = manifest['partitions'][name]
        for p in part['participants']:
            if p in seen_participants:
                raise ValueError(f'Participant {p} appears in {seen_participants[p]} and {name}')
            seen_participants[p] = name
        for r in part['records']:
            if r in seen_records:
                raise ValueError(f'Record {r} appears in {seen_records[r]} and {name}')
            seen_records[r] = name
            if participant_id(r) not in part['participants']:
                raise ValueError(f'Record {r} is split from its participant')
    if set(seen_participants) != set(manifest['participants']):
        raise ValueError('Partition participants differ from the participant table')
    if set(seen_records) != {r for e in manifest['participants'].values() for r in e['records']}:
        raise ValueError('Partition records differ from the participant records')
    for participant, entry in manifest['participants'].items():
        if entry.get('partition') != seen_participants[participant]:
            raise ValueError(f'Participant table assigns {participant} to a different partition')
        if not entry['records'] or any(participant_id(record) != participant for record in entry['records']):
            raise ValueError(f'Participant table contains records owned by another participant: {participant}')
        if any(seen_records[record] != seen_participants[participant] for record in entry['records']):
            raise ValueError(f'Participant table splits records from their owner: {participant}')
    if manifest.get('manifest_sha256') and manifest['manifest_sha256'] != manifest_hash(manifest):
        raise ValueError('Split manifest hash mismatch')
    return True


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    verify_manifest(manifest)
    return manifest


def records_from_inventory(inventory_path, mode='raw', split='train'):
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf-8'))
    rows = []
    for row in inventory['records']:
        if row['mode'] == mode and row['split'] == split:
            counts = row['full_valid_epoch_counts']
            if isinstance(counts, dict):
                counts = [counts['W'] if stage == 'W' and 'W' in counts else
                          counts['Wake'] if stage == 'W' else counts[stage] for stage in STAGES]
            rows.append({'record_id': row['record_id'], 'epoch_counts': counts})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', default='reports/data_inventory.json')
    parser.add_argument('--output', default='reports/split_manifest.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--forced_holdout', default='',
                        help='Explicit comma-separated participants fixed in the holdout.')
    parser.add_argument('--holdout_extra', default='',
                        help='Explicit additional holdout quotas, as cohort=count pairs.')
    parser.add_argument('--validation', default='',
                        help='Explicit validation quotas, as cohort=count pairs.')
    parser.add_argument('--cv_folds', type=int, default=0, help='Write this many cross-validation fold manifests to --cv_dir.')
    parser.add_argument('--cv_dir', default='reports/cv')

    def quotas(text):
        return {k: int(v) for k, v in (item.split('=') for item in text.split(',') if item)}

    args = parser.parse_args()
    if args.cv_folds:
        records = records_from_inventory(args.inventory)
        holdout = load_manifest(args.output)['partitions']['holdout']['participants']
        Path(args.cv_dir).mkdir(parents=True, exist_ok=True)
        for fold in make_cv_folds(records, holdout, args.cv_folds, args.seed):
            Path(args.cv_dir, f"fold_{fold['fold']}.json").write_text(json.dumps(fold, indent=2) + '\n', encoding='ascii')
            print(fold['fold'], fold['partitions']['validation']['participants'], fold['partitions']['validation']['stage_counts'])
        return
    if not (args.forced_holdout or args.holdout_extra):
        parser.error('Specify --forced_holdout or --holdout_extra for an explicit holdout.')
    if not args.validation:
        parser.error('Specify --validation for an explicit development validation partition.')
    manifest = make_split(records_from_inventory(args.inventory), args.seed,
                          [p for p in args.forced_holdout.split(',') if p],
                          quotas(args.holdout_extra), quotas(args.validation))
    if any(not manifest['partitions'][name]['participants'] for name in PARTITIONS):
        parser.error('Explicit split settings must leave nonempty train, validation and holdout partitions.')
    Path(args.output).write_text(json.dumps(manifest, indent=2) + '\n', encoding='ascii')
    print(json.dumps({n: {'participants': len(manifest['partitions'][n]['participants']),
                          'records': len(manifest['partitions'][n]['records']),
                          'epochs': manifest['partitions'][n]['labeled_epochs'],
                          'stage_counts': manifest['partitions'][n]['stage_counts']}
                      for n in PARTITIONS}, indent=2))


if __name__ == '__main__':
    main()
