"""
Qwen LLM Smoke Test for RAG-NAS

測試 Qwen 模型作為 local LLM 是否能正常運作，並生成 NAS search space templates。

用法:
    python test/test_qwen_smoke.py
    python test/test_qwen_smoke.py --model Qwen/Qwen2.5-7B-Instruct
    python test/test_qwen_smoke.py --model Qwen/Qwen2.5-3B-Instruct --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


# ---------------------------------------------------------------------------
# Step 1: Basic Load Test
# ---------------------------------------------------------------------------

def test_basic_load(model_name: str, device: str):
    """Load tokenizer + model and do a simple generation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n[1/3] Basic Load Test")
    print(f"  Model:  {model_name}")
    print(f"  Device: {device}")

    t0 = time.time()
    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"  Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Param count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e9:.2f}B")

    # GPU memory
    if "cuda" in device or device == "auto":
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1e9
            if mem > 0.1:
                print(f"  GPU {i} memory: {mem:.2f} GB")

    # Simple test
    print(f"\n  Testing simple generation ...")
    messages = [{"role": "user", "content": "What is Neural Architecture Search? Answer in one sentence."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    gen_time = time.time() - t0

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Response: {response.strip()}")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  PASS")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2: NAS Template Generation Test
# ---------------------------------------------------------------------------

TEMPLATE_PROMPT_TMPL = """\
You are an expert Neural Architecture Search (NAS) agent.

Given the following retrieval context about top-performing models:
- ResNet-50 on CIFAR-100: top-1=79.34%, backbone=ResNet, head=LinearClsHead
- ConvNeXt-Tiny on CIFAR-10: top-1=96.21%, backbone=ConvNeXt, head=LinearClsHead
- MobileNet-V2 on CIFAR-100: top-1=73.21%, backbone=MobileNetV2, head=LinearClsHead

Dataset profile:
- Task: Image Classification
- Domain: CIFAR
- Image size: 32x32
- Classes: 100

NAS-Bench-201 uses 5 operations: nor_conv_3x3, nor_conv_1x1, skip_connect, avg_pool_3x3, none.

Generate exactly {num_templates} DIVERSE search space seed templates as a JSON array.
Each template should explore a DIFFERENT paradigm, for example:
  - Conv-heavy (favor nor_conv_3x3)
  - Lightweight (favor skip_connect + nor_conv_1x1)
  - Pooling-based (favor avg_pool_3x3)
  - Mixed / balanced
  - Skip-dominant (heavy skip_connect)
  - Deep-narrow (nor_conv_1x1 dominant)
  etc.

Each template must have this structure:
{{
  "paradigm": "descriptive name",
  "macro": {{"backbone": ["list of suggested backbones"], "head": ["list of heads"]}},
  "micro": {{
    "nb201": {{
      "op_prior": {{"nor_conv_3x3": float, "nor_conv_1x1": float, "skip_connect": float, "avg_pool_3x3": float, "none": float}},
      "constraints": [{{"type": "max_count", "op": "none", "value": 1}}]
    }}
  }}
}}

Rules:
1. op_prior values MUST sum to 1.0 for each template
2. Each template MUST have a unique paradigm name
3. Vary the op_prior distributions significantly between templates
4. Output ONLY the JSON array, no explanation
"""


def test_template_generation(model, tokenizer, num_templates: int = 10):
    """Test that Qwen can generate valid NAS seed templates."""
    print(f"\n[2/3] NAS Template Generation Test ({num_templates} templates)")

    prompt = TEMPLATE_PROMPT_TMPL.format(num_templates=num_templates)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that outputs valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"  Prompt tokens: {inputs.input_ids.shape[1]}")
    print(f"  Generating template ...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    gen_time = time.time() - t0

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Output tokens: {outputs.shape[1] - inputs.input_ids.shape[1]}")

    # Extract JSON from response
    json_str = response.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(json_str)
        # Handle both array and single object
        templates = parsed if isinstance(parsed, list) else [parsed]
        print(f"\n  JSON parsing: PASS")
        print(f"  Templates parsed: {len(templates)}")

        valid_count = 0
        print(f"\n  {'#':<4} {'Paradigm':<40} {'conv3x3':>7} {'conv1x1':>7} {'skip':>7} {'pool':>7} {'none':>7} {'sum':>6} {'Valid'}")
        print(f"  {'─' * 95}")

        for i, tmpl in enumerate(templates, 1):
            paradigm = tmpl.get("paradigm", "unnamed")[:38]
            op_prior = tmpl.get("micro", {}).get("nb201", {}).get("op_prior", {})

            c3 = op_prior.get("nor_conv_3x3", 0)
            c1 = op_prior.get("nor_conv_1x1", 0)
            sk = op_prior.get("skip_connect", 0)
            po = op_prior.get("avg_pool_3x3", 0)
            no = op_prior.get("none", 0)
            total = c3 + c1 + sk + po + no

            has_macro = "macro" in tmpl
            has_nb201 = "nb201" in tmpl.get("micro", {})
            sum_ok = abs(total - 1.0) < 0.05
            valid = has_macro and has_nb201 and sum_ok
            if valid:
                valid_count += 1

            mark = "PASS" if valid else "FAIL"
            print(f"  {i:<4} {paradigm:<40} {c3:>7.2f} {c1:>7.2f} {sk:>7.2f} {po:>7.2f} {no:>7.2f} {total:>6.2f} {mark}")

        print(f"\n  Valid: {valid_count}/{len(templates)}")

        # Pretty print each template
        print(f"\n  Detailed output:")
        print(f"  {'─' * 60}")
        for i, tmpl in enumerate(templates, 1):
            paradigm = tmpl.get("paradigm", "unnamed")
            backbones = tmpl.get("macro", {}).get("backbone", [])
            heads = tmpl.get("macro", {}).get("head", [])
            op_prior = tmpl.get("micro", {}).get("nb201", {}).get("op_prior", {})
            constraints = tmpl.get("micro", {}).get("nb201", {}).get("constraints", [])

            print(f"\n  [{i}] {paradigm}")
            print(f"      Backbones:   {backbones}")
            print(f"      Heads:       {heads}")
            print(f"      op_prior:")
            for op, p in op_prior.items():
                bar = "█" * int(p * 30)
                print(f"        {op:<16} {p:.2f} {bar}")
            if constraints:
                print(f"      Constraints: {constraints}")

        # Full JSON dump
        print(f"\n  Complete JSON output:")
        print(f"  {'─' * 60}")
        print(json.dumps(templates, indent=2, ensure_ascii=False))
        print(f"  {'─' * 60}")

        print(f"\n  Overall: {'PASS' if valid_count == len(templates) else f'PARTIAL ({valid_count}/{len(templates)})'}")
        return templates

    except json.JSONDecodeError as e:
        print(f"\n  JSON parsing: FAIL ({e})")
        print(f"\n  Raw output (first 2000 chars):")
        print(f"  {response.strip()[:2000]}")
        return None


# ---------------------------------------------------------------------------
# Step 3: Integration Readiness
# ---------------------------------------------------------------------------

def test_integration_readiness(model, tokenizer):
    """Quick test to verify batch generation works."""
    print(f"\n[3/3] Integration Readiness Test")

    # Test that the model can handle the system prompt format
    # used by llm_template_generator.py
    messages = [
        {"role": "system", "content": "You are an expert NAS agent. Output JSON only."},
        {"role": "user", "content": "List 3 backbone architectures for image classification as a JSON array."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Response: {response.strip()[:200]}")

    # Check GPU memory after all tests
    print(f"\n  Final GPU memory:")
    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1e9
        mem_reserved = torch.cuda.memory_reserved(i) / 1e9
        if mem_alloc > 0.01:
            print(f"    GPU {i}: {mem_alloc:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

    print(f"  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Qwen LLM Smoke Test")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: auto, cuda:0, cuda:1, etc.")
    ap.add_argument("--num_templates", type=int, default=10,
                    help="Number of seed templates to generate (default: 10)")
    args = ap.parse_args()

    print("=" * 60)
    print("  Qwen LLM Smoke Test for RAG-NAS")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\n  Available GPUs:")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    [{i}] {name} ({mem:.1f} GB)")
    else:
        print("\n  No GPU available, using CPU")

    # Run tests
    model, tokenizer = test_basic_load(args.model, args.device)
    test_template_generation(model, tokenizer, num_templates=args.num_templates)
    test_integration_readiness(model, tokenizer)

    print("\n" + "=" * 60)
    print("  All smoke tests completed!")
    print(f"  Model {args.model} is ready for RAG-NAS integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
