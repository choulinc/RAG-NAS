"""
Local LLM Template Generator using Qwen/Qwen2.5-14B-Instruct via HuggingFace Transformers.

This module mirrors the same interface as TemplateGenerator in llm_template_generator.py,
but runs inference locally on GPU using the Qwen2.5-14B-Instruct model.
"""

import json
import re
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Re-use the same prompt and helper from the main module
from src.retrieval.llm_template_generator import SYSTEM_PROMPT, build_context_text


def is_cuda_available() -> bool:
    """Check whether CUDA is available for local model inference."""
    if torch is None:
        return False
    return torch.cuda.is_available()


class LocalTemplateGenerator:
    """Template generator that runs Qwen2.5-14B-Instruct locally on GPU."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

    def __init__(self, model_name: str | None = None, device: str = "auto"):
        if torch is None:
            raise RuntimeError(
                "torch / transformers not installed. "
                "Install with: pip install torch transformers accelerate"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device

        print(f"Loading tokenizer for {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        print(f"Loading model {self.model_name} (this may take a while) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API  (same signature as TemplateGenerator.generate_templates)
    # ------------------------------------------------------------------
    def generate_templates(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        profile=None,
    ) -> List[Dict[str, Any]]:
        context_str = build_context_text(hits)

        # Enrich prompt with dataset profile if available
        profile_str = ""
        if profile:
            profile_str = (
                f"\n--- DATASET PROFILE ---\n"
                f"Task: {profile.task}\n"
                f"Domain: {profile.domain}\n"
                f"Keywords: {', '.join(profile.keywords)}\n"
                f"Number of classes: {profile.num_classes}\n"
                f"Image size (median): {profile.image_stats.median_height}x{profile.image_stats.median_width}\n"
                f"Image channels: {profile.image_stats.channels}\n"
                f"Total images: {profile.image_stats.total_count}\n"
            )

        user_message = (
            f"User Query: {query}\n"
            f"{profile_str}\n"
            f"Context:\n{context_str}\n\n"
            "Please output the JSON object with the 'templates' array."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        print(f"Running local inference with {self.model_name} ...")

        # Use the chat template provided by the tokenizer
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return self._parse_response(content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(content: str) -> List[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try to extract JSON from markdown code block first
        json_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL
        )
        text_to_parse = json_match.group(1).strip() if json_match else content.strip()

        try:
            data = json.loads(text_to_parse)
            if isinstance(data, dict) and "templates" in data:
                raw = data["templates"]
            elif isinstance(data, list):
                raw = data
            else:
                raw = [data]
        except json.JSONDecodeError:
            print("Failed to decode JSON from local LLM response.")
            print(content[:2000])
            return []

        from src.retrieval.llm_template_generator import _validate_template, _print_validation_summary
        validated = [_validate_template(t) for t in raw if isinstance(t, dict)]
        _print_validation_summary(validated)
        return validated
