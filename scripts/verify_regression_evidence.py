#!/usr/bin/env python3
"""Verify collected/executed coverage and source hashes, without fixed audit counts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def tracked_paths(root):
    paths = {root / 'pyproject.toml'}
    for directory in ('src', 'tests', 'scripts', 'config', 'docker'):
        paths.update(p for p in (root / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {str(p.relative_to(root)) for p in paths if p.is_file()}


def verify(data, root):
    collected = data['collected']
    reports = data['reports']
    by_node = {}
    for report in reports:
        by_node.setdefault(report['nodeid'], []).append(report)
    coverage = len(collected) == len(set(collected)) and set(collected) == set(by_node)
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'incomplete': 0}
    for node in collected:
        rows = by_node.get(node, [])
        phases = [row['phase'] for row in rows]
        if len(phases) != len(set(phases)):
            coverage = False
        if any(row['outcome'] == 'failed' for row in rows):
            counts['failed'] += 1
        elif any(row['outcome'] == 'skipped' for row in rows):
            counts['skipped'] += 1
        elif set(phases) == {'setup', 'call', 'teardown'} and all(row['outcome'] == 'passed' for row in rows):
            counts['passed'] += 1
        else:
            counts['incomplete'] += 1
    current_paths = tracked_paths(root)
    hashes = data['source_hashes']
    hash_match = set(hashes) == current_paths and bool(hashes)
    for relative, expected in hashes.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            hash_match = False
            continue
        hash_match = hash_match and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return {'release': data['release'], 'collected': len(collected), **counts,
            'complete_collection_coverage': coverage, 'source_hashes_match': bool(hash_match),
            'all_passed': bool(collected) and coverage and hash_match and not data['collection_errors']
                          and data['exit_code'] == 0 and counts['passed'] == len(collected),
            'native_cfd_qualified': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('verification-local'))
    args = ap.parse_args()
    try:
        data = json.loads((args.results / 'regression.json').read_text())
        summary = verify(data, Path(__file__).resolve().parents[1])
        (args.results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return 0 if summary['all_passed'] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Evidence verification failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
