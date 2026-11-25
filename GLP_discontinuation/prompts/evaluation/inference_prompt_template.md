# Inference Prompt Template for GLP-1 Discontinuation Classification

Use this template for running inference with fine-tuned or base models.

## Quick Reference

### System Message
```
You are a clinical NLP specialist trained to analyze medical documentation and identify reasons for medication discontinuation. You provide accurate, evidence-based classifications in structured JSON format.
```

### User Prompt Template
```
Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond ONLY with a valid JSON object:
{
  "primary_reason": "CATEGORY_NAME",
  "confidence": "high|medium|low",
  "supporting_evidence": "key phrases from note",
  "secondary_factors": []
}

Clinical Note:
{clinical_note}
```

## Python Inference Code

```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_MESSAGE = """You are a clinical NLP specialist trained to analyze medical documentation and identify reasons for medication discontinuation. You provide accurate, evidence-based classifications in structured JSON format."""

INFERENCE_PROMPT = """Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond ONLY with a valid JSON object:
{{
  "primary_reason": "CATEGORY_NAME",
  "confidence": "high|medium|low",
  "supporting_evidence": "key phrases from note",
  "secondary_factors": []
}}

Clinical Note:
{clinical_note}"""


def format_inference_prompt(clinical_note: str, tokenizer) -> str:
    """Format prompt for inference using model's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": INFERENCE_PROMPT.format(clinical_note=clinical_note)}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def extract_json_response(text: str) -> dict:
    """Extract JSON from model response."""
    import re
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    
    # Find JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def run_inference(model, tokenizer, clinical_note: str, max_new_tokens: int = 512):
    """Run inference on a clinical note."""
    prompt = format_inference_prompt(clinical_note, tokenizer)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return extract_json_response(response)


# Example usage:
# model = AutoModelForCausalLM.from_pretrained("your-finetuned-model")
# tokenizer = AutoTokenizer.from_pretrained("your-finetuned-model")
# result = run_inference(model, tokenizer, clinical_note_text)
# print(result)
```

## vLLM Inference Code

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def batch_inference_vllm(model_path: str, clinical_notes: list[str]):
    """Run batch inference using vLLM."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=512
    )
    
    prompts = [
        format_inference_prompt(note, tokenizer) 
        for note in clinical_notes
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        text = output.outputs[0].text
        parsed = extract_json_response(text)
        results.append(parsed)
    
    return results
```

## Expected Output Format

```json
{
  "primary_reason": "ADVERSE_EFFECTS_TOLERABILITY",
  "confidence": "high",
  "supporting_evidence": "persistent nausea, intermittent vomiting, diarrhea temporally related to injections",
  "secondary_factors": []
}
```

## Valid Categories

| Category | Code |
|----------|------|
| Adverse Effects/Tolerability | `ADVERSE_EFFECTS_TOLERABILITY` |
| Access/Cost/System Barriers | `ACCESS_COST_SYSTEM_BARRIERS` |
| Efficacy/Expectation Mismatch | `EFFICACY_EXPECTATION_MISMATCH` |
| Patient Preference/Beliefs | `PATIENT_PREFERENCE_BELIEFS` |
| Planned/Goal-Based Discontinuation | `PLANNED_GOAL_BASED_DISCONTINUATION` |
| Clinical Decision/Medical Appropriateness | `CLINICAL_DECISION_MEDICAL_APPROPRIATENESS` |
| Other/Unknown | `OTHER_UNKNOWN` |

## Confidence Levels

- **high**: Clear, explicit documentation of discontinuation reason
- **medium**: Reason can be inferred but not explicitly stated
- **low**: Multiple possible reasons or ambiguous documentation

## Tips for Best Results

1. **Temperature**: Use low temperature (0.1-0.3) for consistent classifications
2. **JSON Parsing**: Always validate JSON output and handle malformed responses
3. **Batch Processing**: Use vLLM for high-throughput inference
4. **Context Length**: Clinical notes can be long; ensure adequate context window
