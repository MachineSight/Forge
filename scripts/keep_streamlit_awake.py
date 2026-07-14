#!/usr/bin/env python3
"""Keep a Streamlit app warm by sending periodic HTTP pings."""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from urllib import error, request


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ping(url: str, timeout: float, user_agent: str) -> int:
    headers = {"User-Agent": user_agent}

    # Some hosts do not support HEAD reliably, so fall back to GET.
    head_req = request.Request(url, method="HEAD", headers=headers)
    try:
        with request.urlopen(head_req, timeout=timeout) as response:
            return response.getcode()
    except error.HTTPError as exc:
        return exc.code
    except Exception:
        get_req = request.Request(url, method="GET", headers=headers)
        with request.urlopen(get_req, timeout=timeout) as response:
            return response.getcode()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Periodically ping a Streamlit app URL to reduce idle sleep."
    )
    parser.add_argument(
        "--url",
        default="https://forge-team.streamlit.app/",
        help="Target app URL.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=540,
        help="Seconds between pings (default: 540 = 9 minutes).",
    )
    parser.add_argument(
        "--max-jitter-seconds",
        type=int,
        default=20,
        help="Random jitter added to each interval to avoid rigid timing.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default="forge-keepalive/1.0",
        help="HTTP User-Agent header value.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ping once and exit.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Exit after this many consecutive failures. 0 means never exit.",
    )

    args = parser.parse_args()

    if args.interval_seconds <= 0:
        print("interval-seconds must be > 0", file=sys.stderr)
        return 2

    consecutive_failures = 0

    while True:
        try:
            status = ping(args.url, args.timeout_seconds, args.user_agent)
            if 200 <= status < 400:
                print(f"[{utc_now()}] OK {status} - {args.url}")
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                print(
                    f"[{utc_now()}] WARN status={status} failures={consecutive_failures} - {args.url}",
                    file=sys.stderr,
                )
        except Exception as exc:
            consecutive_failures += 1
            print(
                f"[{utc_now()}] ERROR failures={consecutive_failures} - {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        if args.once:
            return 0 if consecutive_failures == 0 else 1

        if args.max_failures > 0 and consecutive_failures >= args.max_failures:
            print(
                f"[{utc_now()}] Exiting after {consecutive_failures} consecutive failures.",
                file=sys.stderr,
            )
            return 1

        jitter = random.uniform(0, max(args.max_jitter_seconds, 0))
        time.sleep(args.interval_seconds + jitter)


if __name__ == "__main__":
    raise SystemExit(main())
