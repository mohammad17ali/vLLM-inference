
# scripts/infer.py
import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from utils import read_prompts, save_jsonl, RESULTS_DIR, rss_mb, Timer

# Try vLLM; if it fails on CPU, use transformers
def load_llm_vllm(model_name: str, max_model_len: int):
    from vllm import LLM
    # For CPU inference, vLLM automatically detects CPU mode
    # Remove device="cpu" as it's not supported in this version
    llm = LLM(model=model_name, trust_remote_code=True, max_model_len=max_model_len)
    print(f"[load_llm_vllm] -Loaded vLLM model '{model_name}' on CPU with max_model_len={max_model_len}")
    return llm

def run_vllm_infer(llm, prompts: List[str], temperature: float, top_p: float, max_tokens: int):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    with Timer() as t:
        outputs = llm.generate(prompts, sampling_params=sp)
    total_time = t.elapsed
    # Parse results
    generations = []
    for out in outputs:
        # vLLM returns multiple candidates per prompt; we take the first
        text = out.outputs[0].text if out.outputs else ""
        generations.append(text)
    return generations, total_time

def run_transformers_infer(model_name: str, prompts: List[str], temperature: float, top_p: float, max_tokens: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.eval()
    gens = []
    with Timer() as t:
        for p in prompts:
            inputs = tok(p, return_tensors="pt")
            # Use typical nucleus sampling on CPU
            out = mdl.generate(
                **inputs,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_tokens
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            gens.append(text)
    total_time = t.elapsed
    return gens, total_time

def main():
    parser = argparse.ArgumentParser(description="Minimal CPU-only inference (vLLM -> transformers fallback)")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name")
    parser.add_argument("--prompts", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "prompts.txt"))
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=1024, help="Context length for vLLM")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts.")
    print(f"Starting RSS memory: {rss_mb():.2f} MB")

    # Attempt vLLM first
    use_vllm = True
    try:
        llm = load_llm_vllm(args.model, args.max_model_len)
    except Exception as e:
        print(f"[WARN] vLLM failed on CPU: {e}\nFalling back to transformers.")
        use_vllm = False
        llm = None

    if use_vllm:
        generations, total_time = run_vllm_infer(llm, prompts, args.temperature, args.top_p, args.max_tokens)
        engine = "vllm"
    else:
        generations, total_time = run_transformers_infer(args.model, prompts, args.temperature, args.top_p, args.max_tokens)
        engine = "transformers"

    # Save outputs
    rows = []
    for prompt, gen in zip(prompts, generations):
        rows.append({"engine": engine, "model": args.model, "prompt": prompt, "output": gen})
    out_path = RESULTS_DIR / "inference_outputs.jsonl"
    save_jsonl(out_path, rows)

    avg_time_per_prompt = total_time / len(prompts)
    print(f"[{engine}] Total time: {total_time:.3f}s | Avg/prompt: {avg_time_per_prompt:.3f}s | RSS: {rss_mb():.2f} MB")
    print(f"Saved generations to {out_path}")

if __name__ == "__main__":
    main()
   
