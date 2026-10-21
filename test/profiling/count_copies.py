#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Count GPU memory copies per execution from a rocprofv3 trace.

Copies large enough for a DMA engine appear in the memory copy trace with
their direction. Small copies run as blit kernels and appear only in the
kernel trace, without a direction. Both are counted.

  rocprofv3 --memory-copy-trace --kernel-trace --output-format csv \\
      -d /tmp/prof -o run -- tritonserver --model-repository ...
  # after load: note the model's execution_count from
  #   curl localhost:8000/v2/models/MODEL/stats
  # run the load, note execution_count again, then stop the server
  count_copies.py /tmp/prof --executions N --skip-s SECONDS_TO_LOAD

If the host-to-device count per execution is close to the number of model
inputs, the execution provider copies each input separately.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict


def read_rows(trace_dir, suffix):
    paths = glob.glob(
        os.path.join(trace_dir, "**", f"*{suffix}.csv"), recursive=True
    )
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_dir", help="rocprofv3 output directory")
    parser.add_argument(
        "--executions",
        type=int,
        default=0,
        help="model executions in the counted window, for per-execution counts",
    )
    parser.add_argument(
        "--skip-s",
        type=float,
        default=0.0,
        help="ignore the first SECONDS of the trace, e.g. load and warmup",
    )
    parser.add_argument(
        "--inputs", type=int, default=0, help="model input count, for comparison"
    )
    parser.add_argument(
        "--blit-prefix",
        default="__amd_rocclr_copy",
        help="kernel name prefix of runtime copy kernels",
    )
    args = parser.parse_args()

    copies, copy_paths = read_rows(args.trace_dir, "memory_copy_trace")
    kernels, kernel_paths = read_rows(args.trace_dir, "kernel_trace")
    if not copy_paths and not kernel_paths:
        sys.exit(f"no memory copy or kernel trace CSV under {args.trace_dir}")

    starts = [int(r["Start_Timestamp"]) for r in copies + kernels]
    if not starts:
        sys.exit("the trace has no records")
    window_start = min(starts) + int(args.skip_s * 1e9)

    # (kind, stream) -> [count, total ns]
    totals = defaultdict(lambda: [0, 0])
    window_end = window_start
    for r in copies:
        start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if start < window_start:
            continue
        kind = r["Direction"].replace("MEMORY_COPY_", "")
        totals[(kind, r["Stream_Id"])][0] += 1
        totals[(kind, r["Stream_Id"])][1] += end - start
        window_end = max(window_end, end)
    kernel_count = 0
    for r in kernels:
        start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if start < window_start:
            continue
        kernel_count += 1
        window_end = max(window_end, end)
        if r["Kernel_Name"].startswith(args.blit_prefix):
            totals[("BLIT_KERNEL", r["Stream_Id"])][0] += 1
            totals[("BLIT_KERNEL", r["Stream_Id"])][1] += end - start

    span_s = (window_end - window_start) / 1e9
    print(f"window: {span_s:.3f} s, {kernel_count} kernels")
    per_exec = args.executions > 0
    header = f"{'kind':<18}{'stream':>8}{'count':>10}{'avg us':>10}"
    if per_exec:
        header += f"{'per exec':>11}{'us/exec':>10}"
    print(header)

    by_kind = defaultdict(int)
    for (kind, stream), (count, ns) in sorted(totals.items()):
        by_kind[kind] += count
        line = f"{kind:<18}{stream:>8}{count:>10}{ns / count / 1e3:>10.1f}"
        if per_exec:
            line += f"{count / args.executions:>11.1f}"
            line += f"{ns / args.executions / 1e3:>10.1f}"
        print(line)

    if per_exec:
        to_gpu = by_kind["HOST_TO_DEVICE"] / args.executions
        blits = by_kind["BLIT_KERNEL"] / args.executions
        print(
            f"\nper execution: {to_gpu:.1f} host-to-device copies, "
            f"{blits:.1f} blit copies of any direction"
        )
        if args.inputs > 0:
            ratio = (to_gpu + blits) / args.inputs
            print(f"model inputs: {args.inputs}, {ratio:.2f} copies per input")


if __name__ == "__main__":
    main()
