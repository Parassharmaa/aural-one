"""Build the 40/30/30 Aural One update pattern from caller-provided manifests.

Every second update includes a same-voice SUBESCO pair. No audio is copied.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PATTERN = ((3, 3, 2), (3, 2, 3), (3, 2, 3), (3, 3, 2), (4, 2, 2))
EMOTIONS = ('angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad')
TYPES = ('choice', 'noul', 'score')


def rows(root, name):
    return [json.loads(line) for line in (root / (name + '.jsonl')).open()]


class GroupCycle:
    def __init__(self, entries, key, seed):
        if not entries:
            raise ValueError('Empty sampling group')
        self.rng = random.Random(seed)
        self.pools = defaultdict(list)
        for entry in entries:
            self.pools[entry[key]].append(entry)
        self.groups = list(self.pools)
        self.rng.shuffle(self.groups)
        for values in self.pools.values():
            self.rng.shuffle(values)
        self.positions = dict.fromkeys(self.groups, 0)
        self.cursor = 0

    def draw(self):
        if self.cursor == len(self.groups):
            self.rng.shuffle(self.groups)
            self.cursor = 0
        group = self.groups[self.cursor]
        self.cursor += 1
        values = self.pools[group]
        position = self.positions[group]
        if position == len(values):
            self.rng.shuffle(values)
            position = 0
        self.positions[group] = position + 1
        return values[position]


def main(args):
    root, output, audit_path, seed = args.data_root, args.output, args.audit, args.seed
    crema, subesco, typed = (rows(root, name) for name in
                             ('crema_train', 'subesco_train', 'typed_train'))
    pairs = rows(root, 'subesco_train_same_sentence_contrasts')
    train_lookup = {row['id']: row for row in subesco}
    if not pairs or len({row['id'] for row in pairs}) != len(pairs):
        raise RuntimeError('Training contrast manifest is empty or has duplicate IDs')
    for pair in pairs:
        left, right = train_lookup[pair['left_id']], train_lookup[pair['right_id']]
        if ((left['speaker_key'], left['sentence_id'], left['trial']) !=
                (right['speaker_key'], right['sentence_id'], right['trial']) or
                left['perceived_consensus_index'] != pair['left_label_index'] or
                right['perceived_consensus_index'] != pair['right_label_index']):
            raise RuntimeError('Invalid training contrast metadata')
    excluded = {row['id'] for name in ('crema_dev', 'subesco_dev', 'typed_dev')
                if (root / (name + '.jsonl')).is_file() for row in rows(root, name)}
    if excluded & {row['id'] for row in crema + subesco + typed}:
        raise RuntimeError('Training/development ID overlap')

    crema_cycles = {}
    for index, emotion in enumerate(EMOTIONS):
        pool = [row for row in crema if row['source_label'] == emotion and
                max(row['target_distribution']) >= 0.5]
        if (len(pool) < args.min_class_rows or
                len({row['actor'] for row in pool}) < args.min_class_actors):
            raise RuntimeError('Insufficient speaker-diverse perceived-label pool: ' + emotion)
        crema_cycles[emotion] = GroupCycle(pool, 'actor', seed + 11 + index)
    subesco_cycle = GroupCycle(subesco, 'speaker_key', seed + 30)
    pair_cycle = GroupCycle(pairs, 'speaker_key', seed + 31)
    typed_cycles = {typ: GroupCycle([row for row in typed if row['type'] == typ],
                                    'group', seed + 40 + index)
                    for index, typ in enumerate(TYPES)}
    rng = random.Random(seed + 50)
    plan = []
    selected = defaultdict(list)
    class_cursor = typed_cursor = 0
    paired_ids = []
    for step in range(1, args.updates + 1):
        crema_count, subesco_count, typed_count = PATTERN[(step - 1) % len(PATTERN)]
        chunks = []
        for _ in range(crema_count):
            emotion = EMOTIONS[class_cursor % len(EMOTIONS)]
            class_cursor += 1
            row = crema_cycles[emotion].draw()
            selected['crema'].append(row)
            chunks.append([{'source': 'crema', 'id': row['id']}])
        if step % 2 == 0:
            pair = pair_cycle.draw()
            paired_ids.append(pair['id'])
            chunks.append([{'source': 'subesco', 'id': pair['left_id'],
                            'pair_id': pair['id'], 'pair_side': 0},
                           {'source': 'subesco', 'id': pair['right_id'],
                            'pair_id': pair['id'], 'pair_side': 1}])
            subesco_count -= 2
        for _ in range(subesco_count):
            row = subesco_cycle.draw()
            selected['subesco'].append(row)
            chunks.append([{'source': 'subesco', 'id': row['id']}])
        for _ in range(typed_count):
            typ = TYPES[typed_cursor % len(TYPES)]
            typed_cursor += 1
            row = typed_cycles[typ].draw()
            selected['typed'].append(row)
            chunks.append([{'source': 'typed', 'type': typ, 'id': row['id']}])
        rng.shuffle(chunks)
        examples = [item for chunk in chunks for item in chunk]
        if len(examples) != 8 or any(item['id'] in excluded for item in examples):
            raise RuntimeError('Wrong Stage-B update size or development leakage')
        plan.append({'step': step, 'examples': examples})
    if output.exists() or audit_path.exists():
        raise FileExistsError('A schedule or audit already exists at this path')
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row) + '\n' for row in plan))
    audit = {'seed': seed, 'steps': len(plan), 'examples_per_step': 8,
             'plan_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
             'pair_manifest_sha256': hashlib.sha256(
                 (root / 'subesco_train_same_sentence_contrasts.jsonl').read_bytes()).hexdigest(),
             'source_exposures': {'crema': len(selected['crema']),
                                  'subesco': sum(item['source'] == 'subesco'
                                                 for update in plan for item in update['examples']),
                                  'typed': len(selected['typed'])},
             'crema_perceived_label_exposures': dict(Counter(row['source_label']
                                                              for row in selected['crema'])),
             'crema_actor_coverage': len({row['actor'] for row in selected['crema']}),
             'paired_updates': len(paired_ids),
             'pair_unique_ids': len(set(paired_ids)),
             'pair_speakers': len({pair['speaker_key'] for pair in pairs
                                    if pair['id'] in set(paired_ids)}),
             'typed_type_exposures': dict(Counter(row['type'] for row in selected['typed'])),
             'development_examples_exposed': 0, 'reserved_test_examples_exposed': 0}
    if args.updates == 1000 and audit['source_exposures'] != \
            {'crema': 3200, 'subesco': 2400, 'typed': 2400}:
        raise RuntimeError('Stage-B source mix changed')
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--min-class-rows', type=int, default=1)
    parser.add_argument('--min-class-actors', type=int, default=1)
    options = parser.parse_args()
    if options.updates <= 0:
        parser.error('--updates must be positive')
    main(options)
