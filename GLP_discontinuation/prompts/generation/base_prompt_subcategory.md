Generate one high-fidelity synthetic clinical note from a VA-style electronic health record for a patient with type 2 diabetes. The output should closely resemble real-world VA documentation while remaining synthetic.

Write the note in a semi-structured clinical style. It may include any realistic combination of:
- subjective history / interval history / HPI
- assessment and plan
- medication list or medication changes
- problem list or comorbidities
- vitals, labs, exam findings, or mental status findings

The note should feel authentic to real clinical documentation:
- use plausible clinical detail, chronology, and internal logic
- allow realistic variation in tone, structure, and length
- sometimes include abbreviations (`Pt`, `BG`, `A1c`, `BID`, etc.), shorthand, or mild imperfections
- sometimes include templated or boilerplate language
- sometimes include unrelated comorbidities, long medication lists, or extra contextual detail

Controlled variation requirements:
- vary note length, section order, and degree of structure
- vary whether the GLP-1 issue is foregrounded or embedded within broader care
- allow occasional minor inconsistencies, updates, or partially conflicting statements that are typical of real EHR notes
- do not make every note read like the same template

Include a GLP-1 medication (for example semaglutide, liraglutide, tirzepatide, dulaglutide, exenatide, etc.). The clinical scenario may vary:
- the GLP-1 may be continued, tapered, increased, held, or discontinued
- symptoms may be related or unrelated to the GLP-1
- another medication may also be changed or stopped

If GLP-1 discontinuation is present, the language may be explicit, indirect, or provider-driven.

The note must be centered around exactly ONE primary reason category for this run, with a specific subcategory focus area.

**PRIMARY_REASON:** `{PRIMARY_REASON}`

**SUBCATEGORY_FOCUS:** `{SUBCATEGORY_FOCUS}`

**Few-shot examples for this category:** `{CATEGORY_FEW_SHOT_EXAMPLES}`

Category instructions:
- The note MUST primarily reflect `{PRIMARY_REASON}`.
- Within this category, focus specifically on: `{SUBCATEGORY_FOCUS}`
- Do NOT mix multiple primary categories.
- Secondary factors may appear, but the dominant underlying reason must match `{PRIMARY_REASON}`.
- The subcategory focus should guide the specific clinical scenario, symptoms, or circumstances described.
- The reason does not need to be directly labeled; it may require clinical inference.
- Use the few-shot examples as guidance for style, framing, and reasoning patterns only. Do not copy wording, structure, or patient details from them.

Subcategory guidance:
- The subcategory focus provides a specific angle or manifestation within the broader category.
- Ensure the clinical narrative authentically represents this subcategory focus.
- For example, if the subcategory is "GI side effects (nausea, vomiting, diarrhea)", the note should prominently feature gastrointestinal symptoms.
- If the subcategory is "insurance loss or denial of coverage", the note should reflect access issues related to insurance.

Output requirements:
- output exactly one valid JSON object
- do not wrap the JSON in markdown fences
- do not provide any text before or after the JSON
- do not include explanations, summaries, bullet-point rationales, or extra metadata outside the JSON
- do not mention that the note is synthetic, generated, fictional, or based on examples

Use this JSON format:
```json
{
  "primary_reason": "{PRIMARY_REASON}",
  "subcategory_focus": "{SUBCATEGORY_FOCUS}",
  "clinical_note": "<full clinical note text>"
}
```

JSON constraints:
- `primary_reason` must exactly match `{PRIMARY_REASON}`
- `subcategory_focus` must exactly match `{SUBCATEGORY_FOCUS}`
- `clinical_note` must contain the full note as a single JSON string, preserving realistic line breaks with escaped newlines when needed
- the clinical note itself should not contain any surrounding commentary or labels outside the note content
