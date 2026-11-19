"""Check SSD replay against a separate Python cache model.

Cases cover duplicate blocks, repeated evictions, prefix gaps, replacements,
write limits and oversized requests. Access and eviction rules count the
current event. Each case starts with empty caches."""

from fractions import Fraction
from pathlib import Path
import json
import struct
import argparse
import hashlib
import shutil
import subprocess
import tempfile

CODE = Path(__file__).resolve().parent
ROOT = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path.write_text(json.dumps(value, indent=2) + "\n")


def invoke(command, folder, seconds, allow_failure=False):
    """Run one tiny case and keep its output until the temporary directory closes."""
    folder.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        list(map(str, command)), capture_output=True, timeout=seconds
    )
    (folder / "stdout.json").write_bytes(result.stdout)
    (folder / "stderr.txt").write_bytes(result.stderr)
    if not allow_failure:
        result.check_returncode()
    return {"returncode": result.returncode}


def physical_equal(native, audit):
    """Require independent counters and cache digests to match."""
    assert audit["status"] == "PASS"
    for key, value in audit.items():
        if key != "schema":
            assert native[key] == value, ("independent replay mismatch", key)


def choose(event, seed, numerator, denominator):
    mask = (1 << 64) - 1
    value = ((event * 0x9E3779B97F4A7C15) & mask) ^ seed ^ 0x5BF03635
    value = ((value ^ (value >> 33)) * 0xFF51AFD7ED558CCD) & mask
    value = ((value ^ (value >> 33)) * 0xC4CEB9FE1A85EC53) & mask
    value ^= value >> 33
    return (value >> 11) * denominator < numerator * (1 << 53)


def reference(rows, times, tokens, start, stop, capacity, budget, rule):
    memory = []
    flash = []
    touches = {}
    evictions = {}
    programs = []
    records = []
    writes = candidates = 0

    def refresh(cache, block):
        cache.remove(block)
        cache.append(block)

    for row, (blocks, now, count) in enumerate(zip(rows, times, tokens)):
        if now < start:
            continue
        if now >= stop:
            break
        distinct = list(dict.fromkeys(blocks))
        bypass = len(distinct) > 4
        cap = (count - 1) // 16
        memory_prefix = prefix = flash_positions = payloads = paid = left = raw = 0
        if not bypass:
            for block in blocks[:cap]:
                if block not in memory:
                    break
                memory_prefix += 1
            loaded = set()
            for block in blocks[:cap]:
                if block in memory:
                    prefix += 1
                elif block in flash:
                    prefix += 1
                    flash_positions += 1
                    if block not in loaded:
                        loaded.add(block)
                        refresh(flash, block)
                        payloads += 1
                else:
                    break
            missing = [b for b in distinct if b not in memory]
            for block in distinct:
                if block in memory:
                    refresh(memory, block)
            while len(memory) + len(missing) > 4:
                block = memory.pop(0)
                event = len(programs)
                left += 1
                evictions[block] = evictions.get(block, 0) + 1
                write = False
                if block in flash:
                    refresh(flash, block)
                else:
                    candidates += 1
                    mode = rule["mode"]
                    k = rule.get("threshold", 0)
                    if mode == "all":
                        wanted = True
                    elif mode == "coin":
                        wanted = choose(event, 17, 1, 3)
                    elif mode == "stride":
                        wanted = candidates % k == 0
                    elif mode == "access_min":
                        wanted = touches[block] >= k
                    elif mode == "eviction_min":
                        wanted = evictions[block] >= k
                    elif mode == "eviction_exact":
                        wanted = evictions[block] == k
                    else:
                        raise ValueError(mode)
                    write = writes < budget and wanted
                    if write:
                        writes += 1
                        paid += 1
                        if len(flash) == capacity:
                            flash.pop(0)
                        flash.append(block)
                programs.append(write)
            for block in missing:
                raw += touches.get(block, 0) > 0
                memory.append(block)
            for block in distinct:
                touches[block] = touches.get(block, 0) + 1
        records.append(
            (
                row,
                now,
                len(blocks),
                memory_prefix,
                prefix,
                flash_positions,
                payloads,
                paid,
                left,
                int(bypass),
                len(blocks) - cap,
                raw,
                count,
            )
        )
    bits = bytearray((len(programs) + 7) // 8)
    for i, value in enumerate(programs):
        bits[i // 8] |= int(value) << (i % 8)
    return bytes(bits), b"".join(struct.pack("<13q", *r) for r in records), writes


def run():
    spec = read(CODE / "test_rules.json")
    root = ROOT / "extended_fixtures"
    root.mkdir(exist_ok=False)
    patterns = [
        [0, 1],
        [2, 3],
        [0, 4],
        [0, 1],
        [5, 6],
        [0, 1],
        [1, 2, 3, 4, 5],
        [7, 2, 2],
        [0, 1],
        [2, 3],
    ]
    rows = [patterns[i % len(patterns)] for i in range(160)]
    times = (
        [-1] + list(range(11)) + [259199999999] + [259200000000 + i for i in range(147)]
    )
    tokens = [16 * len(blocks) + i % 2 for i, blocks in enumerate(rows)]
    assert len(rows) == len(times) == len(tokens)
    for name, fmt, values in [
        ("tiny_nb.i32", "i", list(map(len, rows))),
        ("tiny_ids.i32", "i", sum(rows, [])),
        ("tiny_ts.i64", "q", times),
        ("tokens.i64", "q", tokens),
    ]:
        (root / name).write_bytes(struct.pack("<" + fmt * len(values), *values))
    cases = 0
    activations = {rule["id"]: 0 for rule in spec["rules"]}
    mutation = False
    for start, stop in [(0, 259200000000), (259200000000, 604800000000)]:
        for capacity in [2, 8]:
            for budget in [3, 1000]:
                for rule in spec["rules"]:
                    d = root / str(cases)
                    d.mkdir()
                    common = [
                        "--prefix",
                        root / "tiny",
                        "--tokens",
                        root / "tokens.i64",
                        "--nb",
                        8,
                        "--hbm",
                        4,
                        "--ssd",
                        capacity,
                        "--budget",
                        budget,
                        "--eval-start-us",
                        start,
                        "--stop-us",
                        stop,
                    ]
                    extra = (
                        ["--threshold", rule["threshold"]]
                        if "threshold" in rule
                        else []
                    )
                    if rule["mode"] == "coin":
                        extra += [
                            "--probability-numerator",
                            1,
                            "--probability-denominator",
                            3,
                        ]
                    invoke(
                        [
                            ROOT / "ssd_replay",
                            *common,
                            "--mode",
                            rule["mode"],
                            "--seed",
                            17,
                            "--out-prefix",
                            d / "native",
                            *extra,
                        ],
                        d / "replay",
                        30,
                    )
                    expected_bits, expected_rows, writes = reference(
                        rows, times, tokens, start, stop, capacity, budget, rule
                    )
                    assert (d / "native.programs.bits").read_bytes() == expected_bits, (
                        cases,
                        rule,
                    )
                    assert (d / "native.rows.i64").read_bytes() == expected_rows, (
                        cases,
                        rule,
                    )
                    native = read(d / "replay/stdout.json")
                    assert native["writes"] == writes
                    activations[rule["id"]] += writes
                    audit = [
                        ROOT / "check_replay",
                        *common,
                        "--program-bits",
                        d / "native.programs.bits",
                        "--verify-policy",
                        rule["mode"],
                        "--seed",
                        17,
                        *extra,
                    ]
                    invoke(
                        [*audit, "--out-prefix", d / "audit_native"], d / "audit", 30
                    )
                    physical_equal(native, read(d / "audit/stdout.json"))
                    assert sha(d / "native.rows.i64") == sha(
                        d / "audit_native.rows.i64"
                    )
                    if (
                        not mutation
                        and rule["mode"] == "stride"
                        and native["processed_evictions"]
                    ):
                        bits = bytearray(expected_bits)
                        bits[0] ^= 1
                        (d / "bad.bits").write_bytes(bits)
                        cmd = list(audit)
                        cmd[cmd.index("--program-bits") + 1] = d / "bad.bits"
                        result = invoke(
                            [*cmd, "--out-prefix", d / "bad"],
                            d / "mutation",
                            30,
                            allow_failure=True,
                        )
                        assert result["returncode"] != 0
                        mutation = True
                    cases += 1
    assert (
        cases == 112 and mutation and all(n > 0 for n in activations.values())
    ), activations
    save(
        ROOT / "EXTENDED_FIXTURE_CHECKS.json",
        dict(
            status="PASS",
            cases=cases,
            literal_python_program_and_all_row_fields=True,
            independent_cpp_cache_and_policy=True,
            every_rule_activated=activations,
            wrong_stride_program_rejected=True,
            duplicates_final_token_caps_rewrites_free_copies_and_oversized_bypass=True,
        ),
        exclusive=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rancho-ssd-check-") as directory:
        ROOT = Path(directory)
        for name in ["ssd_replay", "check_replay"]:
            shutil.copyfile(args.build.resolve() / name, ROOT / name)
            (ROOT / name).chmod(0o700)
        run()
        print("PASS: 112 synthetic cases match the Python reference and C++ checker")
