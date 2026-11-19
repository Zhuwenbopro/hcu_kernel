"""Sweep all integer token counts with the installed LightOp fused operator."""

import argparse
import csv
from importlib.metadata import distribution
import json
from pathlib import Path
from statistics import median

import torch

from .python_api import WIDTH, _launcher


BANK_ROWS = 8192
WARMUP = 2
SAMPLES = 11
SEED = 18627


def load_weight(model):
    if model is None:
        return torch.randn(WIDTH, device="cuda", dtype=torch.bfloat16) * 0.1, 1e-6
    from safetensors import safe_open

    key = "model.language_model.layers.0.post_attention_layernorm.weight"
    shards = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(model / shards[key], framework="pt", device="cpu") as shard:
        weight = shard.get_tensor(key).cuda()
    eps = json.loads((model / "config.json").read_text())["text_config"]["rms_norm_eps"]
    assert weight.shape == (WIDTH,) and weight.dtype == torch.bfloat16
    return weight, eps


def measure(runners, residual, original, buffers, start, stop):
    graphs = {}
    for name, run in runners.items():
        residual.copy_(original)
        run(*buffers[0])
        torch.cuda.current_stream().synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for args in buffers:
                run(*args)
        graphs[name] = graph

    names = tuple(graphs)
    samples = {name: [] for name in names}
    for rep in range(WARMUP + SAMPLES):
        shift = rep % len(names)
        for name in names[shift:] + names[:shift]:
            residual.copy_(original)
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            if rep >= WARMUP:
                samples[name].append(start.elapsed_time(stop) * 1000 / len(buffers))
    return {name: round(median(values), 5) for name, values in samples.items()}


@torch.inference_mode()
def sweep(model, max_tokens, output):
    from lightop import op

    installed = distribution("lightop")
    if Path(op.__file__).resolve().parent != Path(installed.locate_file("lightop")).resolve():
        raise RuntimeError("select the installed LightOp package for the baseline")
    gpu = torch.cuda.get_device_properties()
    if gpu.gcnArchName.split(":")[0] != "gfx936":
        raise RuntimeError("the FP32 kernel requires gfx936")
    torch.manual_seed(SEED)
    weight, eps = load_weight(model)
    gamma = (weight.float() + 1).to(torch.bfloat16)
    launch = _launcher()

    def fp32(x, residual, q, scale):
        stream = torch.cuda.current_stream().cuda_stream
        status = launch(x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                        q.data_ptr(), scale.data_ptr(), x.shape[0], eps, stream)
        if status:
            raise RuntimeError(f"HIP launch failed with status {status}")

    def lightop(x, residual, q, scale):
        op.rms_norm_dynamic_per_token_quant_sglang(
            q, x, gamma, scale, eps, residual, False, -1.0, None, None, True, None,
        )

    x = torch.randn((BANK_ROWS, WIDTH), device="cuda", dtype=torch.bfloat16)
    original_residual = torch.randn_like(x)
    residual = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((BANK_ROWS, 1), device=x.device, dtype=torch.float32)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    print(json.dumps({
        "gpu": gpu.name, "arch": gpu.gcnArchName, "pci_bus": gpu.pci_bus_id,
        "lightop": installed.version, "extension": op.__file__, "epsilon": eps,
        "seed": SEED, "warmup": WARMUP, "samples": SAMPLES, "torch": torch.__version__,
        "weight": "Qwen layer 0 post_attention_layernorm" if model else "synthetic BF16",
    }), flush=True)

    runners = {"lightop": lightop, "fp32": fp32}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("tokens", "calls_per_graph", "samples", "lightop_p50_us", "fp32_p50_us", "speedup"))
        for tokens in range(1, max_tokens + 1):
            calls = min(64, BANK_ROWS // tokens)
            rows = calls * tokens
            buffers = list(zip(*(t[:rows].view(calls, tokens, -1).unbind(0)
                                 for t in (x, residual, q, scale)), strict=True))
            latencies = measure(runners, residual[:rows], original_residual[:rows], buffers, start, stop)
            speedup = latencies["lightop"] / latencies["fp32"]
            writer.writerow((tokens, calls, SAMPLES, latencies["lightop"], latencies["fp32"], round(speedup, 6)))
            if tokens % 256 == 0 or tokens == max_tokens:
                file.flush()
                print(f"Measured {tokens}/{max_tokens} token counts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, choices=range(1, 4097), metavar="1..4096", default=4096)
    parser.add_argument("--model", type=Path, help="optional Qwen checkpoint for weight and epsilon")
    args = parser.parse_args()
    sweep(args.model, args.max_tokens, args.output)


if __name__ == "__main__":
    main()
