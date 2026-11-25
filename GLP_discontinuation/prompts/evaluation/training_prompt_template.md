# Training Prompt Template for GLP-1 Discontinuation Classification

This template is designed for fine-tuning language models to classify GLP-1 receptor agonist discontinuation reasons from clinical notes.

## System Message (for chat models)

```
You are a clinical NLP specialist trained to analyze medical documentation and identify reasons for medication discontinuation. You provide accurate, evidence-based classifications in structured JSON format.
```

## Instruction

```
You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)
```

## Input Format

```
{clinical_note}
```

## Output JSON Schema

```json
{
  "primary_reason": "CATEGORY_NAME",
  "confidence": "high|medium|low",
  "supporting_evidence": "Key phrases from the clinical note",
  "secondary_factors": ["optional", "additional", "factors"]
}
```

## Full Training Example Format (Alpaca Style)

```
### Instruction:
You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)

### Input:
[CLINICAL NOTE TEXT HERE]

### Response:
{
  "primary_reason": "ADVERSE_EFFECTS_TOLERABILITY",
  "confidence": "high",
  "supporting_evidence": "persistent nausea, intermittent vomiting, diarrhea temporally related to injections",
  "secondary_factors": []
}
```

## ShareGPT/Chat Format

```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "[INSTRUCTION + CLINICAL NOTE]"
    },
    {
      "from": "gpt",
      "value": "{\"primary_reason\": \"...\", \"confidence\": \"...\", ...}"
    }
  ]
}
```

## Category Definitions

| Category | Description | Key Indicators |
|----------|-------------|----------------|
| ADVERSE_EFFECTS_TOLERABILITY | Discontinuation due to side effects | nausea, vomiting, diarrhea, GI upset, injection site reactions |
| ACCESS_COST_SYSTEM_BARRIERS | Insurance/cost/supply issues | insurance denied, out-of-pocket cost, shortage, prior auth failed |
| EFFICACY_EXPECTATION_MISMATCH | Medication not meeting expectations | plateau, not losing weight, not working, didn't see results |
| PATIENT_PREFERENCE_BELIEFS | Patient choice/beliefs | scared of side effects, read online, stigma, prefer natural |
| PLANNED_GOAL_BASED_DISCONTINUATION | Planned/goal-reached stop | reached goal, tapering off, A1c normalized, planned pregnancy |
| CLINICAL_DECISION_MEDICAL_APPROPRIATENESS | Provider-driven discontinuation | contraindicated, interaction, switch to surgery, kidney disease |
| OTHER_UNKNOWN | Unclear/undocumented reason | no reason given, lost to follow-up, ambiguous |

## Usage Notes

1. **For Training**: Use the Alpaca or ShareGPT format based on your training framework
2. **Dataset Files**:
   - `training_data_instruction.jsonl` - Standard instruction format
   - `training_data_alpaca.jsonl` - Alpaca format with ### markers
   - `training_data_sharegpt.jsonl` - ShareGPT conversation format
3. **Model Compatibility**: Works with Llama, Qwen, Gemma, Mistral, and other instruction-tuned models
