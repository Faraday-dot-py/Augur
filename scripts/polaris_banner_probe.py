import argparse
import csv
import os
import random
import socket
import statistics
import time
from datetime import datetime

HOST = "polaris.hpc.cpp.edu"
PORT = 22
CLIENT_BANNER_TIMEOUT = 15.0


def probe(host, port, connect_timeout, banner_timeout):
    t0 = time.monotonic()
    try:
        s = socket.create_connection((host, port), timeout=connect_timeout)
    except Exception as e:
        return None, None, "", f"connect:{type(e).__name__}"
    t_connect = time.monotonic() - t0
    s.settimeout(banner_timeout)
    buf = b""
    t1 = time.monotonic()
    err = ""
    try:
        while b"\n" not in buf and len(buf) < 512:
            chunk = s.recv(256)
            if not chunk:
                err = "closed_before_banner"
                break
            buf += chunk
    except socket.timeout:
        err = "banner_timeout"
    except Exception as e:
        err = f"recv:{type(e).__name__}"
    t_banner = time.monotonic() - t1
    s.close()
    banner = buf.split(b"\n")[0].decode("ascii", "replace").strip() if buf else ""
    return t_connect, (t_banner if not err else None), banner, err


def run(args):
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "connect_s", "banner_s", "banner", "error"])
            f.flush()
        while True:
            tc, tb, banner, err = probe(HOST, PORT, 15.0, args.banner_timeout)
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        "" if tc is None else f"{tc:.3f}",
                        "" if tb is None else f"{tb:.3f}", banner, err])
            f.flush()
            time.sleep(args.interval * random.uniform(0.8, 1.2))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - m) / d, (c + m) / d


def summarize(args):
    rows = list(csv.DictReader(open(args.out)))
    n = len(rows)
    if n == 0:
        print("no samples")
        return
    bs = [float(r["banner_s"]) for r in rows if r["banner_s"]]
    slow = sum(1 for r in rows if r["error"] or (r["banner_s"] and float(r["banner_s"]) > CLIENT_BANNER_TIMEOUT))
    lo, hi = wilson(slow, n)
    print(f"samples {n}, {rows[0]['ts']} to {rows[-1]['ts']}")
    print(f"banner > {CLIENT_BANNER_TIMEOUT:.0f}s or error: {slow}/{n} = {slow / n:.3%} (95% CI {lo:.3%}-{hi:.3%})")
    if bs:
        q = statistics.quantiles(bs, n=100) if len(bs) >= 2 else [bs[0]] * 99
        print(f"banner latency s: median {statistics.median(bs):.3f}, p90 {q[89]:.3f}, p99 {q[98]:.3f}, max {max(bs):.3f}")
    errs = {}
    for r in rows:
        if r["error"]:
            errs[r["error"]] = errs.get(r["error"], 0) + 1
    print("errors:", errs or "none")
    byhour = {}
    for r in rows:
        h = r["ts"][11:13]
        byhour.setdefault(h, [0, 0])
        byhour[h][1] += 1
        if r["error"] or (r["banner_s"] and float(r["banner_s"]) > CLIENT_BANNER_TIMEOUT):
            byhour[h][0] += 1
    print("slow/err by hour:", {h: f"{a}/{b}" for h, (a, b) in sorted(byhour.items())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/polaris_banner_latency.csv")
    ap.add_argument("--interval", type=float, default=180.0)
    ap.add_argument("--banner-timeout", type=float, default=60.0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
