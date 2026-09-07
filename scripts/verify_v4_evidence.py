#!/usr/bin/env python3
"""Check complete collected-test coverage and audit-to-source/test references.

This verifies regression evidence, not CFD performance or native qualification.
Use --results verification-local after scripts/run_v4_regression.sh.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="verification")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    result_path = Path(args.results)
    if not result_path.is_absolute():
        result_path = root / result_path
    try:
        seen: list[str] = []
        for name in ("legacy_pytest.xml", "v4_pytest.xml"):
            suites = ET.parse(result_path / name).getroot()
            for suite in suites:
                if any(int(suite.get(key, "0")) for key in ("errors", "failures", "skipped")):
                    raise ValueError(f"{name} contains failed, errored or skipped tests.")
                cases = suite.findall("testcase")
                if len(cases) != int(suite.get("tests", "0")):
                    raise ValueError(f"Inconsistent test count in {name}.")
                for case in cases:
                    if case.find("failure") is not None or case.find("error") is not None or case.find("skipped") is not None:
                        raise ValueError("Non-passing testcase in XML.")
                    classname, testname = case.get("classname"), case.get("name")
                    if not classname or not testname:
                        raise ValueError("Testcase has no stable identity.")
                    seen.append(classname.replace(".", "/") + ".py::" + testname)
        collected = [
            x for x in (result_path / "collected_tests.txt").read_text(encoding="utf-8").splitlines()
            if x.startswith("tests/")
        ]
        if len(seen) != len(set(seen)) or set(seen) != set(collected) or len(collected) != len(seen):
            raise ValueError("Collected tests and executed partitions differ, or contain duplicates.")
        audit = json.loads((root / "docs/V4_AUDIT_33.json").read_text(encoding="utf-8"))
        if [x["id"] for x in audit["audit_items"]] != list(range(1, 34)):
            raise ValueError("Audit does not contain exactly items 1 through 33.")
        for item in audit["audit_items"]:
            if not set(item["passed_test_nodeids"]).issubset(seen):
                raise ValueError(f"Unexecuted test referenced by audit item {item['id']}.")
            for relative, expected in item["implementation_files_sha256"].items():
                p = root / relative
                if p.is_symlink() or root not in p.resolve().parents:
                    raise ValueError(f"Unsafe implementation file: {relative}")
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Implementation changed since audit: {relative}")
        print(json.dumps({"passed":len(seen), "audit_items":33,
                          "complete_collection_coverage":True,
                          "audit_source_hashes_match":True,
                          "native_cfd_qualified":False}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        ap.exit(1, f"Evidence verification failed: {exc}\n")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
