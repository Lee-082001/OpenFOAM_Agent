"""Runner-owned POSIX resource setup, then exec without creating a second process."""
from __future__ import annotations
import json
import os
import sys


def main() -> None:
    limits = json.loads(sys.argv[1])
    if os.name == "posix":
        import resource
        if limits.get("cpu_seconds"):
            n = int(limits["cpu_seconds"])
            resource.setrlimit(resource.RLIMIT_CPU, (n, n))
        if limits.get("memory_bytes"):
            n = int(limits["memory_bytes"])
            resource.setrlimit(resource.RLIMIT_AS, (n, n))
        if limits.get("file_bytes"):
            n = int(limits["file_bytes"])
            resource.setrlimit(resource.RLIMIT_FSIZE, (n, n))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    elif limits.get("cpu_seconds") or limits.get("memory_bytes"):
        raise SystemExit("Configured CPU/address-space limits require POSIX; refusing unbounded execution.")
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
