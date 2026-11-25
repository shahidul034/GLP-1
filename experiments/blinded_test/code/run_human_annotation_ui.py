#!/usr/bin/env python3
"""Gradio-based blind annotation UI for label-independence experiment.

Annotators see ONLY:
  - note text
  - 7 category definitions (descriptive, no label hints)
  - instruction to select primary discontinuation reason

They do NOT see:
  - intended label, model predictions, original note_id, category folder, branch info

Each annotator's work is saved independently to:
  data/annotator_{name}_blind.jsonl

Usage:
  python run_human_annotation_ui.py
"""
import json
from pathlib import Path
import datetime
import gradio as gr

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
BLINDED_FILE = DATA_DIR / "blinded_test_notes.jsonl"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]

CATEGORY_DESCRIPTIONS = {
    "ADVERSE_EFFECTS_TOLERABILITY": (
        "Discontinuation primarily due to gastrointestinal side effects "
        "(nausea, vomiting, diarrhea, constipation), systemic symptoms, "
        "serious adverse events, injection site reactions, or other medication-related adverse effects."
    ),
    "ACCESS_COST_SYSTEM_BARRIERS": (
        "Discontinuation primarily due to insurance coverage issues, "
        "high out-of-pocket costs, prior authorization problems, "
        "medication shortages, pharmacy access barriers, or system-level hurdles."
    ),
    "EFFICACY_EXPECTATION_MISMATCH": (
        "Discontinuation primarily because the medication is not working "
        "as expected — inadequate weight loss, insufficient glycemic control, "
        "weight plateau after initial loss, or perceived loss of effectiveness."
    ),
    "PATIENT_PREFERENCE_BELIEFS": (
        "Discontinuation primarily due to patient concerns or beliefs: "
        "fear of long-term side effects, dislike of injections, stigma, "
        "preference for natural remedies, misinformation, or personal choice."
    ),
    "PLANNED_GOAL_BASED_DISCONTINUATION": (
        "Discontinuation because the patient reached a predefined target: "
        "goal weight achieved, glycemic target met, planned treatment "
        "duration completed, or deliberate elective tapering."
    ),
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": (
        "Discontinuation due to a clinician's medical judgment: "
        "contraindications, drug interactions, pregnancy, declining renal "
        "function, frailty/sarcopenia, surgery preparation, or switching "
        "to an alternative therapy."
    ),
    "OTHER_UNKNOWN": (
        "The primary discontinuation reason is unclear, not documented, "
        "lost to follow-up, or does not fit any of the above categories."
    ),
}


def load_blinded_notes():
    records = []
    with open(BLINDED_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_annotations(annotator_name):
    path = DATA_DIR / f"annotator_{annotator_name}_blind.jsonl"
    anns = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    anns[e["blind_id"]] = e
    return anns, path


def save_annotation(annotator_name, annotation):
    existing, path = load_annotations(annotator_name)
    existing[annotation["blind_id"]] = annotation
    with open(path, "w") as f:
        for rec in sorted(existing.values(), key=lambda x: x["blind_id"]):
            f.write(json.dumps(rec) + "\n")
    return len(existing)


def build_status(records, anns):
    total = len(records)
    done = sum(1 for r in records if r["blind_id"] in anns)
    return f"Reviewed: {done}/{total} ({done / total * 100:.1f}%)" if total else "No data"


def format_category_descriptions():
    lines = []
    for cat in CATEGORIES:
        desc = CATEGORY_DESCRIPTIONS[cat]
        lines.append(f"**{cat}**: {desc}")
    return "\n\n".join(lines)


def create_annotation_interface(annotator_name):
    records = load_blinded_notes()
    cat_descriptions = format_category_descriptions()

    def render(idx):
        idx = max(0, min(idx, len(records) - 1))
        anns, _ = load_annotations(annotator_name)
        rec = records[idx]
        ann = anns.get(rec["blind_id"], {})
        status = build_status(records, anns)
        progress = f"Note {idx + 1} of {len(records)}"
        return (
            rec["blind_id"],
            rec["note_text"],
            status,
            progress,
            ann.get("selected_label", None),
            ann.get("supporting_evidence", ""),
            ann.get("confidence", 0.5),
            "",
            idx,
        )

    def save_current(idx, blind_id, selected_label, evidence_text, conf_value):
        if not selected_label:
            return "", "", "", "", None, "", 0.5, "Please select a category before saving.", idx
        annotation = {
            "blind_id": blind_id,
            "selected_label": selected_label,
            "supporting_evidence": evidence_text,
            "confidence": conf_value,
            "annotator": annotator_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        count = save_annotation(annotator_name, annotation)
        anns, _ = load_annotations(annotator_name)
        new_status = build_status(records, anns)
        return blind_id, "", new_status, "", selected_label, evidence_text, conf_value, f"Saved ({count} total)", idx

    def go_prev(idx):
        new_idx = max(0, idx - 1)
        return render(new_idx)

    def go_next(idx):
        new_idx = min(len(records) - 1, idx + 1)
        return render(new_idx)

    def save_and_next(idx, blind_id, selected_label, evidence_text, conf_value):
        kwargs = {
            "blind_id": blind_id,
            "selected_label": selected_label,
            "evidence_text": evidence_text,
            "conf_value": conf_value,
        }
        res = list(save_current(idx, **kwargs))
        new_idx = min(len(records) - 1, idx + 1)
        next_res = list(render(new_idx))
        # Merge: keep status message, go to next note
        res[7] = f"Saved. {next_res[2]}"
        res[2] = next_res[2]
        res[3] = next_res[3]
        res[8] = new_idx
        return tuple(res)

    outputs = [
        gr.Textbox(label="Note ID"),
        gr.Textbox(label="Clinical Note", lines=25, max_lines=35),
        gr.Textbox(label="Progress"),
        gr.Textbox(label="Position"),
        gr.Radio(choices=CATEGORIES, label="Primary Discontinuation Reason"),
        gr.Textbox(label="Supporting Evidence", lines=3, placeholder="Key phrases from the note..."),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Confidence"),
        gr.Textbox(label="Status"),
        gr.State(value=0),
    ]

    with gr.Blocks(title=f"Blind Annotation — {annotator_name}") as app:
        gr.Markdown(f"# GLP-1 Discontinuation — Label-Independence Blind Annotation")
        gr.Markdown(f"**Annotator: {annotator_name}**")

        with gr.Row():
            outputs[2].render()
            outputs[3].render()

        outputs[0].render()

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                outputs[1].render()
            with gr.Column(scale=2):
                gr.Markdown("### Category Definitions")
                gr.Markdown(cat_descriptions)

        gr.Markdown("---")
        gr.Markdown("### Your Annotation")
        outputs[4].render()
        outputs[5].render()
        outputs[6].render()

        outputs[7].render()

        with gr.Row():
            prev_btn = gr.Button("Previous")
            save_btn = gr.Button("Save", variant="primary")
            next_btn = gr.Button("Next")

        # Wire up
        idx_state = outputs[8]
        inputs_for_save = [idx_state, outputs[0], outputs[4], outputs[5], outputs[6]]

        prev_btn.click(fn=go_prev, inputs=[idx_state], outputs=outputs)
        next_btn.click(fn=go_next, inputs=[idx_state], outputs=outputs)
        save_btn.click(fn=save_and_next, inputs=inputs_for_save, outputs=outputs)

        app.load(fn=lambda: render(0), outputs=outputs)

    return app


def create_app():
    with gr.Blocks(title="Blind Annotation — Select Annotator", theme=gr.themes.Soft()) as main_app:
        gr.Markdown("# GLP-1 Discontinuation — Label-Independence Blind Annotation")
        gr.Markdown(
            "This interface presents clinical notes with **no pre-assigned labels**. "
            "Read each note and select the primary discontinuation reason from the definitions provided."
        )
        gr.Markdown("---")

        annotator_dd = gr.Dropdown(
            choices=["Anu", "Parth", "Adiba"],
            label="Select Your Name",
            value=None,
            interactive=True,
        )
        launch_btn = gr.Button("Launch Annotation Interface", variant="primary")
        info = gr.HTML("")

        def launch_annotation(annotator):
            if not annotator:
                return "<p style='color:red;'>Please select your name.</p>"
            import subprocess, sys
            script = Path(__file__).resolve()
            cmd = [sys.executable, str(script), "--annotator", annotator]
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return (
                f"<p style='color:green;'>Annotation UI launched for <strong>{annotator}</strong> "
                f"on port 7865.</p>"
                f"<p>Open <a href='http://localhost:7865' target='_blank'>http://localhost:7865</a> "
                f"in your browser. If another annotator is already using it, please use separate tabs.</p>"
            )

        launch_btn.click(fn=launch_annotation, inputs=[annotator_dd], outputs=[info])

    return main_app


if __name__ == "__main__":
    import sys
    if "--annotator" in sys.argv:
        idx = sys.argv.index("--annotator")
        name = sys.argv[idx + 1]
        app = create_annotation_interface(name)
        app.launch(server_name="0.0.0.0", server_port=7865, share=False)
    else:
        app = create_app()
        app.launch(server_name="0.0.0.0", server_port=7865, share=False)
