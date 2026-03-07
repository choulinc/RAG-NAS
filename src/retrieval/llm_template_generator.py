import os
import sys
import json
import argparse
from typing import List, Dict, Any

try:
    import openai
except ImportError:
    openai = None

from src.retrieval.retrieve import retrieve
from src.retrieval.rag import parse_query_to_filters

SYSTEM_PROMPT = """\
You are an expert Neural Architecture Search (NAS) and Deep Learning agent.
Your objective is to analyze the provided retrieval context regarding top-performing models in the macro-architecture space (OpenMMLab) and extract their core design paradigms. 

Based on this, you must generate MULTIPLE search space templates (seeds) for an Evolutionary Algorithm (EA). 
Each template represents a distinct "Design Paradigm" (e.g., Heavy Convolutional, Lightweight Mobile) and bridges both the MACRO space (OpenMMLab components) and MICRO space (NAS-Bench-201 ops).

You must return EXACTLY a JSON object with a "templates" key containing an array of template objects.
The expected JSON structure per template:
{
  "paradigm": "String (e.g., 'Heavy Convolutional')",
  "task": "String",
  "dataset": "String",
  "evidence": [
    {
      "doc_id": "doc_id from the context",
      "config_repo_path": "path from context",
      "paper_url": "url from context (if available)",
      "why": "Brief explanation of how this model influenced the paradigm."
    }
  ],
  "macro": {
    "backbone": ["Array of string subsets, e.g., 'ResNet', 'ConvNeXt'"],
    "neck": ["Array of neck candidates"],
    "head": ["Array of head candidates"],
    "hparams": { "lr": ["log", 0.0001, 0.05] },
    "constraints": [
       {"type": "forbidden_pair", "source": "ConvNeXt", "target": "VisionTransformerClsHead", "reason": "..."}
    ]
  },
  "micro": {
    "nb201": {
      "allowed_ops": ["nor_conv_3x3", "nor_conv_1x1", "skip_connect", "avg_pool_3x3", "none"],
      "op_prior": {
        "nor_conv_3x3": 0.45,
        "nor_conv_1x1": 0.15,
        "skip_connect": 0.30,
        "avg_pool_3x3": 0.07,
        "none": 0.03
      },
      "edge_prior": {
        "0->1": {"nor_conv_3x3": 0.6, "skip_connect": 0.25, "nor_conv_1x1": 0.1, "avg_pool_3x3": 0.03, "none": 0.02}
      },
      "constraints": [
        {"type": "max_count", "op": "none", "value": 1, "reason": "Avoid network disconnection"},
        {"type": "min_count", "op": "nor_conv_3x3", "value": 1, "reason": "Feature extraction"}
      ]
    }
  }
}

Constraint mapping for NAS-Bench-201:
NAS-Bench-201 has 6 edges connecting 4 nodes in a DAG cell. Each edge selects 1 of 5 operations: nor_conv_3x3, nor_conv_1x1, skip_connect, avg_pool_3x3, none.
Ensure `op_prior` and any `edge_prior` sum exactly to 1.0. 
Ensure constraints prevent invalid empty graphs (too many 'none's).
Be sure to generate at least 2 radically different templates ensuring great exploration diversity!
"""

def build_context_text(hits: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("--- RETRIEVED CONTEXT FROM OPENMMLAB UIR ---")
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] name: {h.get('name')}")
        lines.append(f"    collection: {h.get('collection')}")
        lines.append(f"    task/dataset: {h.get('task')} / {h.get('dataset')}")
        if h.get("metrics"):
            lines.append(f"    metrics: {h.get('metrics')}")
        lines.append(f"    config: {h.get('config_repo_path')}")
        if h.get("paper_url"):
            lines.append(f"    paper: {h.get('paper_url')}")
        lines.append(f"    doc_id: {h.get('doc_id')}")
        lines.append(f"    summary: {h.get('context_text')}")
        lines.append("")
    return "\n".join(lines)


class TemplateGenerator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def generate_templates(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not openai:
            raise RuntimeError("API client not available. Please install openai: pip install openai")
        
        client = openai.OpenAI() # will pick up OPENAI_API_KEY from env
        context_str = build_context_text(hits)
        user_message = f"User Query: {query}\n\nContext:\n{context_str}\n\nPlease output the JSON object with the 'templates' array."

        print(f"Calling LLM ({self.model_name}) to generate EA search space templates...")
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            # Force JSON format if using newest OpenAI models
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            if "templates" in data:
                return data["templates"]
            return [data] # fallback
        except json.JSONDecodeError:
            print("Failed to decode JSON from LLM response.")
            print(content)
            return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uir_path", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--out", type=str, default="data/processed/templates/llm_templates.json")
    args = ap.parse_args()

    # Add project root to sys.path so we can import src.retrieval
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    print(f"Retrieving top {args.topk} contexts for query: '{args.query}'...")
    filters = parse_query_to_filters(args.query)
    hits = retrieve(uir_path=args.uir_path, query=args.query, filters=filters, topk=args.topk)
    
    if not hits:
        print("No retrieval results found.")
        return

    generator = TemplateGenerator(model_name=args.model)
    templates = generator.generate_templates(args.query, hits)
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully generated {len(templates)} templates and saved to {args.out}")

if __name__ == "__main__":
    main()
