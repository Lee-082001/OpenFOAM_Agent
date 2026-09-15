#!/usr/bin/env python3
"""Run the complete pytest collection and preserve truthful, machine-readable evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Evidence:
    def __init__(self):
        self.collected = []
        self.reports = []
        self.collection_errors = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({'nodeid': report.nodeid, 'error': str(report.longrepr)})

    def pytest_runtest_logreport(self, report):
        self.reports.append({'nodeid': report.nodeid, 'phase': report.when,
                             'outcome': report.outcome, 'duration': report.duration,
                             'detail': str(report.longrepr) if report.failed else ''})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'verification-local')
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import pytest
    import openfoam_agent
    evidence = Evidence()
    code = int(pytest.main([str(ROOT / 'tests'), '-q', '--tb=short',
                           f'--junitxml={output / "pytest.xml"}'], plugins=[evidence]))
    from verify_regression_evidence import tracked_paths
    source_hashes = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                     for relative in sorted(tracked_paths(ROOT))}
    payload = {'schema_version': 1, 'release': openfoam_agent.__version__,
               'python': platform.python_version(), 'pytest': pytest.__version__,
               'exit_code': code, 'collected': evidence.collected,
               'reports': evidence.reports, 'collection_errors': evidence.collection_errors,
               'source_hashes': source_hashes, 'native_cfd_qualified': False}
    (output / 'regression.json').write_text(json.dumps(payload, indent=2) + '\n')
    sys.stdout.flush()
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_regression_evidence.py'),
                    '--results', str(output)], check=False)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
