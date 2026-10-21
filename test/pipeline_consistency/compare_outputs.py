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

"""Check that batching and pipelining don't mix up requests' data.

Each request gets its own deterministic inputs. 'record' sends them one
at a time and saves the outputs; 'compare' sends the same requests
concurrently, so they are batched and spread across staging slots, and
checks every request's outputs against the recording.

  compare_outputs.py record  -m MODEL -f ref.npz   # e.g. pipeline_depth 0
  compare_outputs.py compare -m MODEL -f ref.npz   # e.g. pipeline_depth 1
"""

import argparse
import sys
import threading

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype


def make_inputs(meta, index, seed, int_max):
    rng = np.random.default_rng(seed + index)
    arrays = {}
    for inp in meta.inputs:
        shape = [1] + [int(d) for d in inp.shape[1:]]
        if any(d < 0 for d in shape):
            sys.exit(f"input '{inp.name}' has a dynamic shape: {inp.shape}")
        dtype = triton_to_np_dtype(inp.datatype)
        if inp.datatype == "BOOL":
            data = rng.integers(0, 2, size=shape).astype(bool)
        elif np.issubdtype(dtype, np.integer):
            data = rng.integers(0, int_max, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            data = rng.random(size=shape).astype(dtype)
        else:
            sys.exit(f"input '{inp.name}' has unsupported type {inp.datatype}")
        arrays[inp.name] = data
    return arrays


def build_request(meta, arrays):
    inputs = []
    for inp in meta.inputs:
        t = grpcclient.InferInput(inp.name, arrays[inp.name].shape, inp.datatype)
        t.set_data_from_numpy(arrays[inp.name])
        inputs.append(t)
    outputs = [grpcclient.InferRequestedOutput(o.name) for o in meta.outputs]
    return inputs, outputs


def run(client, meta, args, concurrency):
    results = [None] * args.requests
    errors = []
    slots = threading.Semaphore(concurrency)
    done = threading.Event()
    remaining = [args.requests]
    lock = threading.Lock()

    def callback(index, result, error):
        if error is not None:
            errors.append((index, str(error)))
        else:
            results[index] = {o.name: result.as_numpy(o.name) for o in meta.outputs}
        slots.release()
        with lock:
            remaining[0] -= 1
            if remaining[0] == 0:
                done.set()

    for i in range(args.requests):
        inputs, outputs = build_request(
            meta, make_inputs(meta, i, args.seed, args.int_max)
        )
        slots.acquire()
        client.async_infer(
            args.model,
            inputs,
            outputs=outputs,
            callback=lambda result, error, i=i: callback(i, result, error),
        )
    done.wait()
    if errors:
        for index, message in errors[:5]:
            print(f"request {index} failed: {message}")
        sys.exit(f"{len(errors)} of {args.requests} requests failed")
    return results


def has_nan(value):
    return np.issubdtype(value.dtype, np.floating) and bool(np.isnan(value).any())


def record(results, path):
    flat = {
        f"{i}/{name}": value
        for i, outputs in enumerate(results)
        for name, value in outputs.items()
    }
    nan_requests = sorted({int(k.split("/")[0]) for k, v in flat.items() if has_nan(v)})
    if nan_requests:
        sys.exit(
            f"{len(nan_requests)} of {len(results)} requests returned NaN "
            f"(first: {nan_requests[:5]}); not saving an unusable reference"
        )
    np.savez_compressed(path, **flat)
    print(f"recorded {len(results)} requests to {path}")


def compare(results, path, rtol, atol):
    ref = np.load(path)
    bad = []
    worst = 0.0
    for i, outputs in enumerate(results):
        for name, value in outputs.items():
            key = f"{i}/{name}"
            if key not in ref:
                sys.exit(f"{path} has no entry for {key}; record with the same -n")
            expected = ref[key]
            if value.shape != expected.shape:
                bad.append((i, name, f"shape {value.shape} vs {expected.shape}"))
                continue
            if has_nan(value):
                bad.append((i, name, "NaN in output"))
                continue
            if np.issubdtype(expected.dtype, np.floating):
                diff = float(np.max(np.abs(value - expected), initial=0.0))
                worst = max(worst, diff)
                if not np.allclose(value, expected, rtol=rtol, atol=atol):
                    bad.append((i, name, f"max abs diff {diff:.3g}"))
            elif not np.array_equal(value, expected):
                bad.append((i, name, "values differ"))

    print(f"compared {len(results)} requests, worst float diff {worst:.3g}")
    if bad:
        requests = sorted({i for i, _, _ in bad})
        for i, name, why in bad[:10]:
            print(f"  request {i} output '{name}': {why}")
        sys.exit(f"{len(requests)} of {len(results)} requests mismatched")
    print("all outputs match")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["record", "compare"])
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-f", "--file", required=True, help="reference .npz")
    parser.add_argument("-u", "--url", default="localhost:8001")
    parser.add_argument("-n", "--requests", type=int, default=512)
    parser.add_argument("-c", "--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--int-max",
        type=int,
        default=2,
        help="integer inputs are drawn from [0, int-max); keep it within "
        "every embedding table's size",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    client = grpcclient.InferenceServerClient(args.url)
    meta = client.get_model_metadata(args.model)
    if args.mode == "record":
        record(run(client, meta, args, concurrency=1), args.file)
    else:
        compare(run(client, meta, args, args.concurrency), args.file, args.rtol, args.atol)


if __name__ == "__main__":
    main()
