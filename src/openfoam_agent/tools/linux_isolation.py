"""Operator-owned strict Linux isolation; unavailable guarantees fail closed.

bwrap supplies mount/PID/user/IPC/network namespaces. cgroup v2 applies aggregate
memory, swap, CPU-rate and task caps to wrapper, MPI launcher and all descendants.
A separately provisioned finite filesystem bounds persistent output, including
many small files (RLIMIT_FSIZE alone does not). No privileged mount or delegation
is attempted here. CPU-time and cumulative wall budgets are monitored with a
bounded polling interval; this is explicitly not a real-time deadline guarantee.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from pydantic import BaseModel, ConfigDict, Field
from openfoam_agent.contracts.models import ResourceLimits


class IsolationUnavailable(RuntimeError): pass


class LinuxIsolationPolicy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    mode: str = "strict_linux"
    cgroup_parent: str
    writable_volume: str
    readonly_runtime_roots: list[str] = Field(default_factory=list)
    bwrap_path: str = "/usr/bin/bwrap"
    limits: ResourceLimits

    @classmethod
    def read(cls,path):
        path=Path(path)
        if path.is_symlink() or path.stat().st_size>100000:
            raise IsolationUnavailable("Isolation policy must be a bounded regular operator file.")
        return cls.model_validate_json(path.read_text())


def mount_entries():
    records=[]
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left,right=line.split(" - ",1)
        columns=left.split();kind=right.split()[0]
        path=columns[4].replace("\\040"," ").replace("\\011","\t").replace("\\134","\\")
        records.append((Path(path),kind,columns[5]))
    return records


def bounded_tree_bytes(root):
    """Logical bytes include sparse files; symlinks/special files are rejected."""
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(directory)/name
            if p.is_symlink(): raise IsolationUnavailable("Symlinks are not allowed in native writable storage.")
            if p.is_file(): total+=p.stat().st_size
            elif not p.is_dir(): raise IsolationUnavailable("Special files are not allowed in native storage.")
    return total


@contextmanager
def workspace_execution_lock(root):
    if os.name!="posix":
        yield;return
    import fcntl
    path=Path(root)/".native-execution.lock"
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IsolationUnavailable("Another native process owns this workspace; concurrent execution is forbidden.") from exc
        yield
    finally:
        os.close(fd)


class LinuxIsolation:
    def __init__(self,policy,workspace_root):
        self.policy=policy
        self.workspace=Path(workspace_root).resolve()
        self.volume=Path(policy.writable_volume).resolve()
        self.parent=Path(policy.cgroup_parent).resolve()
        self.bwrap=Path(policy.bwrap_path)
        self.active=None
        self.preflight()

    def preflight(self):
        p=self.policy
        if p.mode!="strict_linux" or os.name!="posix" or not Path("/proc/self/mountinfo").exists():
            raise IsolationUnavailable("Strict isolation requires Linux; no local fallback is permitted.")
        if os.geteuid()==0:
            raise IsolationUnavailable("Run the agent unprivileged; root execution is not an isolation mode.")
        if not self.bwrap.is_file() or not os.access(self.bwrap,os.X_OK) or self.bwrap.resolve().parent not in {Path("/usr/bin"),Path("/bin")}:
            raise IsolationUnavailable("A trusted system bubblewrap executable is required.")
        if self.workspace!=self.volume and self.volume not in self.workspace.parents:
            raise IsolationUnavailable("Workspace is outside the operator-provisioned bounded volume.")
        mounts=mount_entries()
        if not any(path==self.volume and kind in {"tmpfs","ext4","xfs","btrfs"} for path,kind,opts in mounts):
            raise IsolationUnavailable("Writable volume must be a separate bounded filesystem mount, not an ordinary directory.")
        stat=os.statvfs(self.volume)
        if stat.f_blocks*stat.f_frsize>p.limits.max_case_bytes:
            raise IsolationUnavailable("Filesystem capacity exceeds aggregate case quota; a small per-file limit is insufficient.")
        if any(self.volume in path.parents for path,kind,opts in mounts):
            raise IsolationUnavailable("Nested mounts could escape the aggregate writable-volume quota.")
        if not any(kind=="cgroup2" and (path==self.parent or path in self.parent.parents) for path,kind,opts in mounts):
            raise IsolationUnavailable("cgroup_parent is not a delegated cgroup v2 filesystem path.")
        if self.parent==Path("/sys/fs/cgroup"):
            raise IsolationUnavailable("Use an operator-delegated cgroup, never the system cgroup root.")
        enabled=set((self.parent/"cgroup.subtree_control").read_text().split())
        if not {"cpu","memory","pids"}.issubset(enabled):
            raise IsolationUnavailable("The delegated cgroup lacks enabled cpu/memory/pids controllers.")
        if p.limits.memory_bytes is None or p.limits.total_cpu_seconds is None:
            raise IsolationUnavailable("Strict isolation requires aggregate memory and total CPU-time limits.")
        if any(Path(root).resolve() in {Path('/'),Path('/home'),Path('/root'),Path('/etc'),Path('/run'),Path('/proc'),Path('/sys'),self.workspace,self.volume} for root in p.readonly_runtime_roots):
            raise IsolationUnavailable("Broad/private host roots must not be exposed to the sandbox.")
        bounded_tree_bytes(self.workspace)

    def start(self,limits):
        self.preflight()
        if self.active is not None: raise IsolationUnavailable("Nested isolated native invocations are forbidden.")
        group=self.parent/("openfoam-agent-"+uuid.uuid4().hex)
        group.mkdir(mode=0o700)
        try:
            settings={"memory.max":str(limits.memory_bytes),"memory.swap.max":"0","memory.oom.group":"1",
                "pids.max":str(limits.max_processes),"cpu.max":f"{int(limits.cpu_cores*100000)} 100000"}
            for name,value in settings.items():
                target=group/name
                if not target.exists(): raise IsolationUnavailable(f"Required cgroup control is unavailable: {name}")
                target.write_text(value)
            if not (group/"cgroup.kill").exists(): raise IsolationUnavailable("cgroup.kill is required for whole-tree cancellation.")
            self.active=group
        except BaseException:
            group.rmdir();raise
        return str(group)

    def command(self,argv,cwd,env):
        case=Path(cwd).resolve()
        if case!=self.workspace/"case":
            raise IsolationUnavailable("Isolated native execution must use the workspace case directory.")
        scratch=self.workspace/"sandbox-scratch"
        scratch.mkdir(exist_ok=True,mode=0o700)
        tmp=scratch/"tmp";shm=scratch/"shm"
        tmp.mkdir(exist_ok=True);shm.mkdir(exist_ok=True)
        args=[str(self.bwrap),"--die-with-parent","--new-session","--unshare-user","--disable-userns",
              "--unshare-pid","--unshare-net","--unshare-ipc","--unshare-uts","--unshare-cgroup",
              "--cap-drop","ALL","--clearenv"]
        roots=["/usr","/bin","/lib","/lib64",*self.policy.readonly_runtime_roots]
        emitted=set()
        for raw in roots:
            root=Path(raw)
            if not root.exists():
                if raw in self.policy.readonly_runtime_roots: raise IsolationUnavailable("Readonly runtime root is missing.")
                continue
            if str(root) in emitted: continue
            emitted.add(str(root));args += ["--ro-bind",str(root),str(root)]
        for name in ("ld.so.cache","passwd","group","nsswitch.conf","hosts"):
            path=Path("/etc")/name
            if path.is_file(): args += ["--ro-bind",str(path),str(path)]
        args += ["--proc","/proc","--remount-ro","/proc","--dev","/dev",
            "--bind",str(tmp),"/tmp","--bind",str(shm),"/dev/shm","--remount-ro","/dev",
            "--bind",str(case),str(case),"--chdir",str(case)]
        safe=dict(env);safe.update(HOME="/nonexistent",TMPDIR="/tmp")
        for key,value in sorted(safe.items()): args += ["--setenv",key,value]
        args += ["--remount-ro","/","--",*argv]
        return args

    def metrics(self):
        if self.active is None:return {}
        def pairs(name):
            return {line.split()[0]:int(line.split()[1]) for line in (self.active/name).read_text().splitlines()}
        cpu=pairs("cpu.stat");events=pairs("memory.events")
        return {"cpu_seconds":cpu.get("usage_usec",0)/1e6,"memory_current":int((self.active/"memory.current").read_text()),
            "oom_kill":events.get("oom_kill",0),"pids_current":int((self.active/"pids.current").read_text())}

    def finish(self):
        if self.active is None:return {}
        group=self.active
        metrics=self.metrics()
        (group/"cgroup.kill").write_text("1")
        deadline=time.monotonic()+2
        while time.monotonic()<deadline and "populated 1" in (group/"cgroup.events").read_text(): time.sleep(.02)
        empty="populated 1" not in (group/"cgroup.events").read_text()
        metrics.update(self.metrics());metrics["process_tree_empty"]=empty
        if empty: group.rmdir()
        self.active=None
        if not empty: raise IsolationUnavailable("Cgroup process tree did not become empty; manual reconciliation is required.")
        return metrics

    def fingerprint(self):
        return hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest()
