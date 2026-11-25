"""
Human evaluation interface for Gemma-4-31B-it classification results on VA clinical notes.
Launches a Gradio UI to review predictions and save evaluator judgments.
"""

import json
from pathlib import Path
import gradio as gr
import datetime

DATA_FILE = Path(__file__).parent / "classified_va_notes.jsonl"
EVAL_FILE = Path(__file__).parent / "human_evaluations.jsonl"

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]


def load_data():
    records = []
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_evaluations():
    evals = {}
    if EVAL_FILE.exists():
        with open(EVAL_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    evals[e["note_id"]] = e
    return evals


def save_evaluation(eval_record):
    existing = load_evaluations()
    existing[eval_record["note_id"]] = eval_record
    with open(EVAL_FILE, "w") as f:
        for rec in existing.values():
            f.write(json.dumps(rec) + "\n")


def build_status_bar(records, evals):
    total = len(records)
    done = sum(1 for r in records if r["note_id"] in evals)
    return f"Reviewed: {done}/{total}"


def render_note(idx, records, evals):
    if not records:
        return ("No data", "", "", "", "", "0/0", "", gr.update(), gr.update(), gr.update(), gr.update())

    idx = max(0, min(idx, len(records) - 1))
    r = records[idx]
    e = evals.get(r["note_id"], {})

    status_bar = build_status_bar(records, evals)
    progress = f"Note {idx + 1} of {len(records)}"

    # Determine initial radio value
    verdict = e.get("verdict", None)
    corrected_label = e.get("corrected_label", "")
    comments = e.get("comments", "")

    true_label = r.get("primary_reason", "N/A")
    pred_label = r.get("gemma_predicted_reason", "N/A")
    rationale = r.get("gemma_rationale", "")

    # Highlight matching labels
    match_note = "Labels match" if true_label == pred_label else "Labels DIFFER"

    eval_status = ""
    if e:
        eval_status = f"[EVALUATED] Verdict: {e.get('verdict','')}"
        if corrected_label:
            eval_status += f" | Corrected to: {corrected_label}"

    return (
        r["clinical_note"],
        true_label,
        f"{pred_label}",
        rationale,
        match_note,
        status_bar,
        progress,
        gr.update(value=verdict),
        gr.update(value=corrected_label),
        gr.update(value=comments),
        gr.update(value=idx),
        eval_status,
    )


def submit_evaluation(note_idx, verdict, corrected_label, comments, records):
    if not records:
        return "No records", gr.update()

    note_idx = int(note_idx)
    r = records[note_idx]

    eval_record = {
        "note_id": r["note_id"],
        "verdict": verdict,
        "corrected_label": corrected_label if verdict == "Incorrect" else "",
        "comments": comments,
        "timestamp": datetime.datetime.now().isoformat(),
        "true_label": r.get("primary_reason", ""),
        "predicted_label": r.get("gemma_predicted_reason", ""),
    }
    save_evaluation(eval_record)

    evals = load_evaluations()
    status = build_status_bar(records, evals)
    label_status = "Saved."
    return label_status, status


def create_app():
    records = load_data()
    evals = load_evaluations()

    with gr.Blocks(title="GLP-1 Discontinuation – Human Evaluation") as app:
        gr.Markdown("# GLP-1 Discontinuation Reason Classification — Human Evaluation")
        gr.Markdown(
            "Review each clinical note and the Gemma-4-31B-it model's predicted reason "
            "and rationale. Rate the prediction and optionally provide a corrected label."
        )

        with gr.Row():
            status_bar = gr.Textbox(value=build_status_bar(records, evals), label="Progress", interactive=False, scale=2)
            progress_label = gr.Textbox(value=f"Note 1 of {len(records)}", label="Position", interactive=False, scale=1)

        eval_status_text = gr.Textbox(value="", label="Evaluation Status", interactive=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                clinical_note = gr.Textbox(
                    label="Clinical Note",
                    lines=22,
                    max_lines=30,
                    interactive=False,
                )

            with gr.Column(scale=2):
                true_label = gr.Textbox(label="True Label (from data)", interactive=False)
                pred_label = gr.Textbox(label="Gemma Predicted Label", interactive=False)
                match_note = gr.Textbox(label="Label Match", interactive=False)
                rationale = gr.Textbox(
                    label="Gemma Rationale",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                )

        gr.Markdown("---")
        gr.Markdown("### Your Evaluation")

        with gr.Row():
            verdict = gr.Radio(
                choices=["Correct", "Partially Correct", "Incorrect"],
                label="Verdict",
                value=None,
                interactive=True,
            )
            corrected_label = gr.Dropdown(
                choices=[""] + CATEGORIES,
                label="Corrected Label (if Incorrect)",
                value="",
                interactive=True,
            )

        comments = gr.Textbox(
            label="Comments / Notes",
            lines=3,
            placeholder="Optional notes about this prediction...",
            interactive=True,
        )

        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            save_btn = gr.Button("Save & Next", variant="primary")

        note_idx = gr.State(value=0)

        outputs = [
            clinical_note,
            true_label,
            pred_label,
            rationale,
            match_note,
            status_bar,
            progress_label,
            verdict,
            corrected_label,
            comments,
            note_idx,
            eval_status_text,
        ]

        def go_prev(idx, records):
            new_idx = max(0, idx - 1)
            return render_note(new_idx, records, load_evaluations())

        def go_next(idx, records):
            new_idx = min(len(records) - 1, idx + 1)
            return render_note(new_idx, records, load_evaluations())

        def go_to(idx_str, records, direction):
            try:
                idx = int(idx_str)
            except (ValueError, TypeError):
                idx = 0
            if direction == "prev":
                idx = max(0, idx - 1)
            else:
                idx = min(len(records) - 1, idx + 1)
            return render_note(idx, records, load_evaluations())

        def save_and_next(note_idx, verdict, corrected_label, comments):
            note_idx = int(note_idx) if note_idx else 0
            status_msg, new_status = submit_evaluation(note_idx, verdict, corrected_label, comments, records)
            new_idx = min(len(records) - 1, note_idx + 1)
            rendered = list(render_note(new_idx, records, load_evaluations()))
            rendered[-5] = gr.update(value=new_status)
            rendered[-1] = status_msg
            return rendered

        def save_current(note_idx, verdict, corrected_label, comments):
            note_idx = int(note_idx) if note_idx else 0
            status_msg, new_status = submit_evaluation(note_idx, verdict, corrected_label, comments, records)
            return status_msg, new_status

        prev_btn.click(
            fn=go_prev,
            inputs=[note_idx, gr.State(records)],
            outputs=outputs,
        )
        next_btn.click(
            fn=go_next,
            inputs=[note_idx, gr.State(records)],
            outputs=outputs,
        )
        save_btn.click(
            fn=save_and_next,
            inputs=[note_idx, verdict, corrected_label, comments],
            outputs=outputs,
        )

        # Load first note on startup
        app.load(
            fn=lambda: render_note(0, records, load_evaluations()),
            outputs=outputs,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7864, share=False, theme=gr.themes.Soft())
