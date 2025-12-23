
# scripts/benchmark.py
import argparse
import math
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from utils import read_prompts, append_csv, RESULTS_DIR, rss_mb, Timer

def load_vllm(model_name: str, max_model_len: int):
    from vllm import LLM
    return LLM(model=model_name, trust_remote_code=True, device="cpu", max_model_len=max_model_len)

def gen_vllm(llm, prompts: List[str], temperature: float, top_p: float, max_tokens: int):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    with Timer() as t:
        outputs = llm.generate(prompts, sampling_params=sp)
    lat = t.elapsed
    return outputs, lat

def load_tf(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

def gen_tf(tok, mdl, prompts: List[str], temperature: float, top_p: float, max_tokens: int):
    import torch
    start = time.perf_counter()
    for p in prompts:
        inputs = tok(p, return_tensors="pt")
        _ = mdl.generate(
            **inputs,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
    lat = time.perf_counter() - start
    return lat

def main():
    parser = argparse.ArgumentParser(description="CPU benchmarking for vLLM (fallback to transformers)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompts", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "prompts.txt"))
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=1024)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    all_prompts = read_prompts(args.prompts)
    print(f"Starting RSS memory: {rss_mb():.2f} MB")
    # Attempt vLLM
    engine = "vllm"
    llm = None
    try:
        llm = load_vllm(args.model, args.max_model_len)
    except Exception as e:
        print(f"[WARN] vLLM failed on CPU: {e}\nFalling back to transformers.")
        engine = "transformers"
        tok, mdl = load_tf(args.model)

    headers = ["engine", "model", "batch_size", "num_prompts", "total_time_s", "avg_latency_s", "throughput_qps", "rss_mb"]
    bench_path = RESULTS_DIR / "benchmarks.csv"

    for bs in args.batch_sizes:
        # Prepare batches
        num_prompts = len(all_prompts)
        batches = [all_prompts[i:i+bs] for i in range(0, num_prompts, bs)]

        total_time = 0.0
        print(f"\nRunning batch size = {bs} ({engine})")
        for batch in tqdm(batches):
            if engine == "vllm":
                _, lat = gen_vllm(llm, batch, args.temperature, args.top_p, args.max_tokens)
            else:
                lat = gen_tf(tok, mdl, batch, args.temperature, args.top_p, args.max_tokens)
            total_time += lat

        avg_latency = total_time / len(batches)
        throughput = num_prompts / total_time if total_time > 0 else 0.0
        mem = rss_mb()
        append_csv(bench_path, headers, [engine, args.model, bs, num_prompts, round(total_time, 4), round(avg_latency, 4), round(throughput, 4), round(mem, 2)])

        print(f"Batch {bs}: total_time={total_time:.3f}s, avg_latency={avg_latency:.3f}s, throughput={throughput:.2f} qps, RSS={mem:.2f} MB")

    print(f"\nSaved benchmarks to {bench_path}")

if __name__ == "__main__":
    main()