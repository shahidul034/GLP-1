#!/usr/bin/env python3
"""
Detect GLP-1 drug discontinuation from free text using vLLM.
Classifies into one of 7 categories with explanation if discontinuation is found.
Otherwise reports no discontinuation detected.

Usage:
  echo "text" | python3 glp_discontinuation_detector.py
  python3 glp_discontinuation_detector.py --text "Patient stopped Ozempic due to nausea"
  python3 glp_discontinuation_detector.py --file note.txt
  python3 glp_discontinuation_detector.py --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

GLP1_DRUGS = [
    "semaglutide", "ozempic", "wegovy", "rybelsus",
    "liraglutide", "victoza", "saxenda",
    "tirzepatide", "mounjaro", "zepbound",
    "dulaglutide", "trulicity",
    "exenatide", "byetta", "bydureon",
    "glp-1", "glp1", "glp 1",
    "glp-1 ra", "glp1 agonist", "glp-1 agonist",
]

DISCONTINUATION_TERMS = [
    "discontinue", "discontinuation", "stopped", "stop", "quit",
    "cease", "held", "hold", "withdrew", "withdraw",
    "no longer taking", "not taking", "stopped taking",
    "ran out", "expired", "inactive", "deprescribe",
    "taper", "tapered", "tapering", "wean", "weaned",
]

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]

INSTRUCTION = """You are a clinical NLP specialist. Analyze the following clinical note and determine if a GLP-1 receptor agonist was discontinued.

If YES, classify the primary reason into exactly one of these categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects (nausea, vomiting, diarrhea), systemic symptoms, serious adverse events (pancreatitis, gallbladder), injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

If NO discontinuation of GLP-1 is present, respond with: {{"has_discontinuation": false}}

Respond with ONLY a JSON object containing:
1. "has_discontinuation": true or false
2. "primary_reason": category (if true)
3. "explanation": brief explanation of why this category was chosen (if true)
4. "supporting_evidence": key phrases from the text (if true)

DO NOT show your thinking. Output ONLY the JSON."""


def has_glp1_drug(text):
    text_lower = text.lower()
    for drug in GLP1_DRUGS:
        if drug in text_lower:
            return True
    return False


def has_discontinuation(text):
    text_lower = text.lower()
    for term in DISCONTINUATION_TERMS:
        if term in text_lower:
            return True
    return False


class VLLMClassifier:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.85):
        self.model_id = model_id
        self._loaded = False
        self.llm = None
        self.tokenizer = None
        self.gpu_memory_utilization = gpu_memory_utilization

    def load(self):
        if self._loaded:
            return
        from vllm import LLM
        self.llm = LLM(
            model=self.model_id,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=8192,
            tensor_parallel_size=1,
            dtype="auto",
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._loaded = True

    def unload(self):
        if not self._loaded:
            return
        import gc
        import torch
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False

    def build_prompt(self, text):
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Clinical Note:\n{text}"},
        ]
        if self.tokenizer.chat_template is None:
            prompt = "<|begin_of_text|>"
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return prompt
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def parse_output(self, raw_output):
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', cleaned, flags=re.DOTALL)
        json_match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if not json_match:
            return None
        try:
            result = json.loads(json_match.group())
            return result
        except json.JSONDecodeError:
            return None

    def classify(self, text):
        self.load()
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.1, max_tokens=512, top_p=0.9)
        prompt = self.build_prompt(text)
        outputs = self.llm.generate([prompt], params)
        raw = outputs[0].outputs[0].text
        result = self.parse_output(raw)
        return result


def main():
    parser = argparse.ArgumentParser(description="Detect GLP-1 discontinuation from text using vLLM")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", "-t", type=str, help="Text to analyze")
    input_group.add_argument("--file", "-f", type=str, help="Path to input file")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="vLLM model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--gpu-memory", type=float, default=0.85,
                        help="GPU memory utilization (default: 0.85)")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty-print output")
    parser.add_argument("--no-precheck", action="store_true",
                        help="Skip keyword precheck and always use LLM")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print(json.dumps({"error": "No input text provided."}))
        sys.exit(1)

    if not args.no_precheck:
        if not has_glp1_drug(text) or not has_discontinuation(text):
            return

    clf = VLLMClassifier(model_id=args.model, gpu_memory_utilization=args.gpu_memory)
    try:
        clf.load()
        result = clf.classify(text)
        if result is not None and result.get("has_discontinuation") is True:
            print(json.dumps(result, indent=2 if args.pretty else None))
    finally:
        clf.unload()


if __name__ == "__main__":
    main()
