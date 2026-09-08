#!/bin/sh
# Run only Python/mock and synthetic-process regression tests.
# No live model or genuine OpenFOAM invocation is authorized by this wrapper.
set -eu
cd "$(dirname "$0")/.."
OUT="${1:-verification-local}"
mkdir -p "$OUT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q tests \
  --ignore=tests/test_v400_audit_domains.py \
  --ignore=tests/test_v400_execution_contracts.py \
  --ignore=tests/test_v400_release_edges.py \
  --tb=short --junitxml="$OUT/legacy_pytest.xml" > "$OUT/legacy_pytest.txt" 2>&1
cat "$OUT/legacy_pytest.txt"
python -m pytest -q \
  tests/test_v400_audit_domains.py \
  tests/test_v400_execution_contracts.py \
  tests/test_v400_release_edges.py \
  --tb=short --junitxml="$OUT/v4_pytest.xml" > "$OUT/v4_pytest.txt" 2>&1
cat "$OUT/v4_pytest.txt"
python -m pytest --collect-only -q tests > "$OUT/collected_tests.txt"
python scripts/verify_v4_evidence.py --results "$OUT"
