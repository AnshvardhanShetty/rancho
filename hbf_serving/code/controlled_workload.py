"""Make synthetic chats with shared prompt blocks and unique endings"""

import hashlib, struct
from features import RequestHistory
from synthetic_workload import FIT


def tokens(setup, i, calibration=False, warm=False):
    """Build one synthetic prompt with a shared beginning and a unique ending."""
    g = i % setup["groups"]
    p = setup["prefix_tokens"]
    u = setup["unique_suffix_tokens"]
    rep = setup.get("repetition", 17)
    base = 4000 if calibration else 1000
    prefix = [base + g * 64 + j % 32 for j in range(p)]
    tail = 10000 + i if calibration else 30000 + g if warm else 20000 + rep * 1300 + i
    return prefix + [tail] * u


def prompts(setup, calibration=False):
    """Build warm-up prompts followed by the measured prompts."""
    if calibration:
        return [tokens(setup, i, True) for i in range(setup["calibration_requests"])]
    return [tokens(setup, g, warm=True) for g in range(setup["groups"])] + [
        tokens(setup, i) for i in range(setup["requests"])
    ]


def records(setup):
    """Build request features from synthetic earlier session history."""
    h = RequestHistory()
    maxn = setup["max_reference_requests"]
    cal = setup["calibration_requests"]
    gap = setup["logical_gap_seconds"] * 1000000
    cut = FIT["cut_us"]
    row = 0
    for i in range(maxn):
        h.arrival(
            row,
            cut - (maxn + cal - i) * gap,
            i % setup["groups"],
            setup["synthetic_user"],
        )
        row += 1
    earlier = []
    for i in range(cal):
        t = cut - (cal - i) * gap
        info = h.arrival(row, t, 10000 + i, setup["synthetic_user"])
        row += 1
        earlier.append(
            dict(
                sequence_row=row - 1,
                session=10000 + i,
                policy_us=t,
                features=info.shared(),
            )
        )
    measured = []
    for i in range(setup["groups"] + setup["requests"]):
        g = i if i < setup["groups"] else (i - setup["groups"]) % setup["groups"]
        t = cut + i * gap
        info = h.arrival(row, t, g, setup["synthetic_user"])
        row += 1
        measured.append(
            dict(
                sequence_row=row - 1,
                session=g,
                policy_us=t,
                features=info.shared(),
                warmup=i < setup["groups"],
            )
        )
    return earlier, measured


def bound_requests(setup):
    """Associate each engine request ID with its features and session."""
    return {
        f"row{i}": dict(
            features=v["features"], policy_us=v["policy_us"], session=v["session"]
        )
        for i, v in enumerate(records(setup)[1])
    }


def block_keys(prompt):
    """Assign content-based IDs to full pages, including their preceding context."""
    if len(prompt) % 16:
        raise ValueError("Only full input blocks in this controlled study")
    previous = bytes(32)
    out = []
    for a in range(0, len(prompt), 16):
        previous = hashlib.sha256(
            previous + struct.pack("<16I", *prompt[a : a + 16])
        ).digest()
        out.append(previous)
    return out


def eligibility(setup):
    # Reconstruct independent offline-style reference stamps before any cache fill.
    """Mark repeated reads using the independent synthetic reference history."""
    stamps = {}
    for i in range(setup["max_reference_requests"]):
        for b in block_keys(tokens(setup, i)):
            stamps[b] = i % setup["groups"]
    for i in range(setup["calibration_requests"]):
        for b in block_keys(tokens(setup, i, calibration=True)):
            stamps[b] = 10000 + i
    out = []
    for i, prompt in enumerate(prompts(setup)):
        group = i if i < setup["groups"] else (i - setup["groups"]) % setup["groups"]
        keys = block_keys(prompt)
        seen = set()
        bits = []
        for b in keys:
            bits.append(b not in seen and stamps.get(b) == group)
            seen.add(b)
            stamps[b] = group
        out.append(bits)
    return out
