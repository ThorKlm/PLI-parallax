#!/usr/bin/env python
"""Fetch ColabFold MSAs for Boltz inputs, populating the Boltz MSA cache, with no prediction.

Why this exists
---------------
Before committing GPU time to a multimer re-fold we need to know whether the MSA
fetch itself is feasible at scale, and at what rate it has to be throttled.  That
question is separable from prediction: ``boltz predict --use_msa_server`` does the
fetch in ``process_input`` *before* any model is loaded, and the artefacts it leaves
behind are reused verbatim on a later run.  So the fetch can be run on its own, on a
box with no GPU, and the prediction run can consume the result.

What "the Boltz cache" is here
------------------------------
Two different things are called the boltz cache and only one of them holds MSAs:

  * ``--cache`` / ``BOLTZ_CACHE`` (``/workspace/boltz_cache_x/.boltz``) holds model
    weights and the CCD.  Nothing to do with MSAs.
  * ``<out_dir>/boltz_results_<input_dir_name>/msa/`` is the MSA cache
    (``boltz/main.py:745``, ``main.py:1134``).  That is what this script fills.

Inside it, two layers matter and they are *not* equivalent:

  * ``<target_id>_<entity_id>.csv`` -- the parsed MSA Boltz actually reads.
  * ``<target_id>_unpaired_tmp_env/out.tar.gz`` and
    ``<target_id>_paired_tmp_pairgreedy-env/out.tar.gz`` -- the raw server response.

Only the *second* prevents a refetch.  ``process_input`` rebuilds its ``to_generate``
list unconditionally and never checks whether the ``.csv`` already exists
(``boltz/main.py:568-579``); the short-circuit is one level down, in
``run_mmseqs2``'s ``if not os.path.isfile(tar_gz_file)``
(``boltz/data/msa/mmseqs2.py:184``).  Writing only the ``.csv`` files would look like
a populated cache and still refetch everything.  This script therefore goes through
``compute_msa``'s own code path, which produces both.

Request accounting
------------------
Boltz groups chains into entities by ``(entity_type, sequence)`` with the sequence
compared as an **exact string** (``boltz/data/parse/schema.py:1037``), and submits one
MSA per *entity*, not per chain (``main.py:570-574``: ``to_generate`` is keyed by
``f"{target_id}_{entity_id}"``).  So N identical chains cost one request, and a
hetero-N-mer costs N.  Across systems there is no sharing at all: the key is prefixed
with ``target_id``, so the same sequence in two systems is fetched twice.

Per system the server sees at most two jobs, not one per sequence: one unpaired job
carrying every distinct sequence, plus -- only when there are >= 2 distinct protein
entities -- one paired job carrying the same list (``main.py:468-494``).

Backoff accounting
------------------
``run_mmseqs2`` logs every wait as ``Sleeping for <n>s. Reason: <status>``, and the
reasons mean opposite things:

  * ``RATELIMIT`` -- the server is refusing to accept the submission.  This is the
    only line that measures throttling, and the only one that gets worse under load.
  * ``PENDING`` / ``RUNNING`` -- the job was accepted and we are waiting for it.  This
    is queue and compute time.  It scales with how many sequences the job carries, so
    multimers show more of it than monomers whether or not anything is throttled.

A regex that matches ``Sleeping for (\\d+)s`` without splitting on the reason adds
these together and reports queue time as rate limiting.  They are counted separately
here.

A hard refusal is different from both: ``MMseqs2 API is giving errors``,
``MAINTENANCE``, or five consecutive transport failures, each of which
``run_mmseqs2`` raises rather than retries.  That aborts the run.

Usage
-----
    msa_prefetch.py plan  --input-dir DIR --manifest M.tsv --systems s1,s2,...
    msa_prefetch.py run   --manifest M.tsv --input-dir DIR --msa-dir DIR [--limit N]
    msa_prefetch.py report --manifest M.tsv --log L.jsonl

``run`` is resumable: the manifest carries a per-system ``status`` column, is rewritten
after every system, and a rerun picks up only the rows still ``pending``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _Loader

MANIFEST_COLS = ["system_id", "n_chains", "n_entities", "n_residues", "stratum",
                 "status", "attempts"]

# Reasons run_mmseqs2 sleeps on.  RATELIMIT is the server refusing us; the rest is
# waiting for a job it already accepted.  Kept apart on purpose -- see module docstring.
# A transient server hiccup -- a truncated / non-gzip download, a dropped connection --
# surfaces as "error" and is indistinguishable at the call site from a genuinely bad
# input.  Empirically these clear on an immediate retry, so treating "error" as
# terminal quietly leaves holes in the cache.  cleanup_partial() has already removed
# the half-written tmp dir by then, so a retry is a clean refetch.
RETRYABLE_STATUSES = {"error", "timeout"}

# A per-system line is too fine to read across thousands of systems, so a rollup is
# emitted on this interval as well.
PROGRESS_EVERY = 250

SLEEP_RE = re.compile(r"Sleeping for (\d+)s\. Reason: (\w+)")
THROTTLE_REASONS = {"RATELIMIT"}
QUEUE_REASONS = {"PENDING", "RUNNING", "UNKNOWN"}
NET_RETRY_RE = re.compile(r"Error while fetching result from MSA server")

# Raised through by run_mmseqs2 rather than retried: we are not being served.
HARD_REFUSAL_RE = re.compile(
    r"MMseqs2 API is giving errors|undergoing maintenance|"
    r"Too many failed attempts|Too many jobs|quota exceeded|429",
    re.I,
)


# --------------------------------------------------------------------------- parsing


def parse_entities(yaml_path: Path):
    """Reproduce Boltz's entity grouping for one input.

    Mirrors ``parse_boltz_schema`` (``boltz/data/parse/schema.py:1015-1090``): items are
    grouped by ``(entity_type, sequence)`` in first-appearance order and the entity id
    is the index into that grouping -- counting ligands, which is why a protein that
    follows a ligand does not get index 0.  Getting this wrong would write the MSA csv
    under a filename Boltz does not look for.

    Returns (n_chains, {msa_key_suffix: sequence}) for the protein entities only.
    """
    with yaml_path.open() as fh:
        schema = yaml.load(fh, Loader=_Loader)

    items_to_group: dict[tuple[str, str], list] = {}
    for item in schema["sequences"]:
        etype = next(iter(item)).lower()
        if etype in {"protein", "dna", "rna"}:
            seq = str(item[etype]["sequence"])
        elif etype == "ligand":
            seq = str(item[etype].get("smiles", item[etype].get("ccd")))
        else:
            raise ValueError(f"invalid entity type {etype} in {yaml_path}")
        items_to_group.setdefault((etype, seq), []).append(item)

    entities: dict[int, str] = {}
    for entity_id, ((etype, seq), items) in enumerate(items_to_group.items()):
        if etype != "protein":
            continue
        # An explicit `msa:` key means Boltz never calls the server for that entity
        # (main.py:570 only queues chains whose msa_id is still 0).  boltz_in_mc carries
        # `msa: empty`; boltz_in_mc_msa does not.
        if any("msa" in it["protein"] for it in items):
            continue
        entities[entity_id] = seq

    total_chains = 0
    for item in schema["sequences"]:
        etype = next(iter(item)).lower()
        if etype != "protein":
            continue
        ids = item[etype]["id"]
        total_chains += 1 if isinstance(ids, str) else len(ids)

    return total_chains, entities


# ---------------------------------------------------------------------------- worker


class SleepAccountant(logging.Handler):
    """Reads the sleep reasons out of run_mmseqs2's own log records."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.throttle_s = 0
        self.throttle_n = 0
        self.queue_s = 0
        self.queue_n = 0
        self.other_s = 0
        self.net_retries = 0
        self.lines: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        m = SLEEP_RE.search(msg)
        if m:
            secs, reason = int(m.group(1)), m.group(2).upper()
            if reason in THROTTLE_REASONS:
                self.throttle_s += secs
                self.throttle_n += 1
            elif reason in QUEUE_REASONS:
                self.queue_s += secs
                self.queue_n += 1
            else:
                self.other_s += secs
            self.lines.append(msg)
        elif NET_RETRY_RE.search(msg):
            self.net_retries += 1
            self.lines.append(msg)


def fetch_one(yaml_path: Path, msa_dir: Path, target_id: str, server: str,
              pairing: str) -> dict:
    """Fetch the MSAs for one system into ``msa_dir``.

    The body below is ``boltz.main.compute_msa`` (``boltz/main.py:468-524``) inlined,
    not reimplemented: same two ``run_mmseqs2`` calls, same prefixes, same truncation
    constants, same csv layout.  Inlining it rather than importing ``boltz.main`` keeps
    torch out of the process -- which matters when this is spawned 300 times -- and
    makes room for the instrumentation.  Any drift from upstream here would silently
    produce a cache Boltz refuses to reuse, so it is worth re-diffing after a boltz
    upgrade.
    """
    from boltz.data import const
    from boltz.data.msa import mmseqs2 as mm

    acct = SleepAccountant()
    mm.logger.addHandler(acct)
    mm.logger.setLevel(logging.DEBUG)

    n_chains, entities = parse_entities(yaml_path)
    data = {f"{target_id}_{eid}": seq for eid, seq in entities.items()}

    unpaired_tar = msa_dir / f"{target_id}_unpaired_tmp_env" / "out.tar.gz"
    paired_tar = msa_dir / f"{target_id}_paired_tmp_pairgreedy-env" / "out.tar.gz"
    was_cached = unpaired_tar.exists() and (len(data) < 2 or paired_tar.exists())

    rec = {
        "system_id": target_id,
        "n_chains": n_chains,
        "n_entities": len(data),
        "n_residues": sum(len(s) for s in data.values()),
        "longest_seq": max((len(s) for s in data.values()), default=0),
        "paired_job": len(data) > 1,
        "n_jobs": (2 if len(data) > 1 else 1) if data else 0,
        "already_cached": was_cached,
    }

    t0 = time.time()
    try:
        if not data:
            rec.update(status="skipped", note="no protein entity needing an MSA")
        else:
            if len(data) > 1:
                paired_msas = mm.run_mmseqs2(
                    list(data.values()),
                    msa_dir / f"{target_id}_paired_tmp",
                    use_env=True,
                    use_pairing=True,
                    host_url=server,
                    pairing_strategy=pairing,
                )
            else:
                paired_msas = [""] * len(data)

            unpaired_msa = mm.run_mmseqs2(
                list(data.values()),
                msa_dir / f"{target_id}_unpaired_tmp",
                use_env=True,
                use_pairing=False,
                host_url=server,
                pairing_strategy=pairing,
            )

            depths = []
            for idx, name in enumerate(data):
                paired = paired_msas[idx].strip().splitlines()
                paired = paired[1::2]
                paired = paired[: const.max_paired_seqs]

                keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
                paired = [s for s in paired if s != "-" * len(s)]

                unpaired = unpaired_msa[idx].strip().splitlines()
                unpaired = unpaired[1::2]
                unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
                if paired:
                    unpaired = unpaired[1:]

                seqs = paired + unpaired
                keys = keys + [-1] * len(unpaired)

                csv_str = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, seqs)]
                (msa_dir / f"{name}.csv").write_text("\n".join(csv_str))
                depths.append(len(seqs))

            rec.update(status="ok", msa_depth=depths)
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        rec.update(
            status="refused" if HARD_REFUSAL_RE.search(text) else "error",
            error=text,
        )
    finally:
        mm.logger.removeHandler(acct)

    rec.update(
        wall_s=round(time.time() - t0, 1),
        throttle_s=acct.throttle_s,
        throttle_n=acct.throttle_n,
        queue_s=acct.queue_s,
        queue_n=acct.queue_n,
        other_sleep_s=acct.other_s,
        net_retries=acct.net_retries,
        sleep_lines=acct.lines[-40:],
    )
    return rec


# ---------------------------------------------------------------------------- driver


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    """Rewrite atomically -- a manifest truncated by a kill is an unresumable run."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write("\t".join(MANIFEST_COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in MANIFEST_COLS) + "\n")
    os.replace(tmp, path)


def cmd_plan(a: argparse.Namespace) -> int:
    systems = [s.strip() for s in Path(a.systems).read_text().split() if s.strip()]
    in_dir = Path(a.input_dir)
    rows = []
    for sid in systems:
        n_chains, entities = parse_entities(in_dir / f"{sid}.yaml")
        rows.append(dict(system_id=sid, n_chains=n_chains, n_entities=len(entities),
                         n_residues=sum(len(q) for q in entities.values()),
                         stratum=stratum_of(n_chains), status="pending", attempts=0))
    write_manifest(Path(a.manifest), rows)
    print(f"planned {len(rows)} systems -> {a.manifest}")
    return 0


def stratum_of(n: int) -> str:
    for lo, hi in [(2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return "129+"


def cleanup_partial(msa_dir: Path, sid: str) -> None:
    """Drop half-written tmp dirs.

    A tarball truncated by a timeout or a kill still satisfies run_mmseqs2's
    ``os.path.isfile`` check, so it would be treated as a cache hit forever and the
    system would silently get a corrupt MSA.  Removing the whole tmp dir forces a clean
    refetch on the next attempt.
    """
    for suffix in ("_unpaired_tmp_env", "_paired_tmp_pairgreedy-env"):
        d = msa_dir / f"{sid}{suffix}"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def cmd_run(a: argparse.Namespace) -> int:
    manifest = Path(a.manifest)
    in_dir = Path(a.input_dir)
    msa_dir = Path(a.msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log)

    rows = read_manifest(manifest)
    by_id = {r["system_id"]: r for r in rows}

    def select_todo():
        """Rows still owed a fetch: never attempted, or failed transiently and not yet
        out of attempts.  Recomputed per pass so retries are picked up automatically
        instead of needing an operator to reset the manifest by hand."""
        out = []
        for r in rows:
            st = r["status"]
            if st == "pending":
                out.append(r)
            elif st in RETRYABLE_STATUSES and int(r.get("attempts") or 0) < a.max_attempts:
                out.append(r)
        return out

    done_before = sum(1 for r in rows if r["status"] == "ok")
    print(f"[prefetch] {len(rows)} in manifest, {done_before} already ok, "
          f"{len(select_todo())} to fetch (<={a.max_attempts} attempts each)", flush=True)

    cum_throttle = 0
    cum_queue = 0
    t_start = time.time()
    n_done = 0
    refusals = []

    for pass_no in range(1, a.max_attempts + 1):
        todo = select_todo()
        if a.limit:
            todo = todo[: a.limit]
        if not todo:
            break
        if pass_no > 1:
            print(f"[prefetch] retry pass {pass_no}: re-attempting {len(todo)} system(s) "
                  f"after {a.retry_backoff:.0f}s backoff", flush=True)
            time.sleep(a.retry_backoff)

        for i, row in enumerate(todo, 1):
            sid = row["system_id"]
            out_json = msa_dir / f".{sid}.rec.json"
            cmd = [sys.executable, os.path.abspath(__file__), "worker",
                   "--yaml", str(in_dir / f"{sid}.yaml"), "--msa-dir", str(msa_dir),
                   "--system-id", sid, "--out", str(out_json),
                   "--server", a.server, "--pairing", a.pairing]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=a.per_system_timeout)
                rc = proc.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                rc, timed_out = -1, True
                proc = None

            if timed_out:
                cleanup_partial(msa_dir, sid)
                rec = dict(system_id=sid, status="timeout",
                           wall_s=round(time.time() - t0, 1),
                           n_chains=int(row["n_chains"]), n_entities=int(row["n_entities"]),
                           throttle_s=0, queue_s=0,
                           note=f"exceeded {a.per_system_timeout}s")
            elif out_json.exists():
                rec = json.loads(out_json.read_text())
                out_json.unlink()
            else:
                rec = dict(system_id=sid, status="error",
                           wall_s=round(time.time() - t0, 1),
                           n_chains=int(row["n_chains"]), n_entities=int(row["n_entities"]),
                           throttle_s=0, queue_s=0,
                           error=(proc.stderr[-2000:] if proc else "no record written"))

            if rec["status"] not in ("ok", "skipped"):
                cleanup_partial(msa_dir, sid)

            rec["run_index"] = done_before + n_done + 1
            rec["attempt"] = int(by_id[sid].get("attempts") or 0) + 1
            rec["pass"] = pass_no
            rec["elapsed_s"] = round(time.time() - t_start, 1)
            cum_throttle += rec.get("throttle_s", 0)
            cum_queue += rec.get("queue_s", 0)
            rec["cum_throttle_s"] = cum_throttle
            rec["cum_queue_s"] = cum_queue

            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

            by_id[sid]["status"] = rec["status"]
            by_id[sid]["attempts"] = rec["attempt"]
            write_manifest(manifest, rows)
            n_done += 1

            print(f"[{i}/{len(todo)}] p{pass_no} {sid} chains={rec.get('n_chains')} "
                  f"entities={rec.get('n_entities')} {rec['status']} "
                  f"{rec.get('wall_s')}s throttle={rec.get('throttle_s',0)}s "
                  f"queue={rec.get('queue_s',0)}s | cum throttle {cum_throttle}s "
                  f"queue {cum_queue}s | elapsed {rec['elapsed_s']:.0f}s", flush=True)

            if i % a.progress_every == 0 or i == len(todo):
                n_ok = sum(1 for r in rows if r["status"] == "ok")
                n_bad = sum(1 for r in rows
                            if r["status"] not in ("ok", "skipped", "pending", ""))
                el = time.time() - t_start
                rate = el / max(n_done, 1)
                left = (len(todo) - i) * rate
                print(f"[progress] pass {pass_no} {i}/{len(todo)} | "
                      f"ok={n_ok}/{len(rows)} problems={n_bad} | "
                      f"elapsed {el/3600:.2f}h | {rate:.1f}s/system | "
                      f"eta {left/3600:.1f}h | cum throttle {cum_throttle}s "
                      f"queue {cum_queue}s", flush=True)

            if rec["status"] == "refused":
                refusals.append(time.time())
                recent = [t for t in refusals if time.time() - t < 600]
                print(f"[prefetch] refusal on {sid} ({len(recent)} in last 10min): "
                      f"{rec.get('error')}", flush=True)
                if len(recent) >= 3:
                    print("[prefetch] STOP: 3 refusals within 10 minutes, "
                          "service likely down", flush=True)
                    return 2
            if cum_throttle > a.throttle_budget:
                print(f"[prefetch] STOP: cumulative throttling {cum_throttle}s exceeds "
                      f"budget {a.throttle_budget}s", flush=True)
                return 3
            if a.sleep_between:
                time.sleep(a.sleep_between)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    stuck = [r for r in rows if r["status"] not in ("ok", "skipped", "pending")]
    print(f"[prefetch] finished {n_done} fetches in {time.time()-t_start:.0f}s | "
          f"ok={n_ok}/{len(rows)} | unresolved={len(stuck)}", flush=True)
    for r in stuck[:50]:
        print(f"[prefetch]   unresolved {r['system_id']} status={r['status']} "
              f"attempts={r['attempts']}", flush=True)
    return 0



def cmd_report(a: argparse.Namespace) -> int:
    recs = [json.loads(l) for l in Path(a.log).read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    walls = sorted(r["wall_s"] for r in ok)
    print(json.dumps(dict(
        n=len(recs), n_ok=len(ok),
        statuses={s: sum(1 for r in recs if r["status"] == s) for s in
                  {r["status"] for r in recs}},
        wall_total_s=round(sum(r["wall_s"] for r in recs), 1),
        mean_s=round(statistics.mean(walls), 1) if walls else 0,
        median_s=round(statistics.median(walls), 1) if walls else 0,
        cum_throttle_s=sum(r.get("throttle_s", 0) for r in recs),
        cum_queue_s=sum(r.get("queue_s", 0) for r in recs),
        chains=sum(r.get("n_chains", 0) for r in recs),
        entities=sum(r.get("n_entities", 0) for r in recs),
    ), indent=2))
    return 0


def cmd_worker(a: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.CRITICAL)
    rec = fetch_one(Path(a.yaml), Path(a.msa_dir), a.system_id, a.server, a.pairing)
    Path(a.out).write_text(json.dumps(rec))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--systems", required=True, help="file of system ids")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--msa-dir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--server", default="https://api.colabfold.com")
    p.add_argument("--pairing", default="greedy")
    p.add_argument("--per-system-timeout", type=int, default=900)
    p.add_argument("--throttle-budget", type=int, default=1800,
                   help="cumulative RATELIMIT seconds after which the run stops")
    p.add_argument("--sleep-between", type=float, default=0.0)
    p.add_argument("--progress-every", type=int, default=PROGRESS_EVERY,
                   help="emit a rollup line every N systems")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="attempts per system before it is left unresolved")
    p.add_argument("--retry-backoff", type=float, default=60.0,
                   help="seconds to pause before each retry pass")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("worker")
    p.add_argument("--yaml", required=True)
    p.add_argument("--msa-dir", required=True)
    p.add_argument("--system-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--server", default="https://api.colabfold.com")
    p.add_argument("--pairing", default="greedy")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("report")
    p.add_argument("--manifest", required=True)
    p.add_argument("--log", required=True)
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
