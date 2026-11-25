import json
import os
import re
from datetime import datetime

import gradio as gr

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_JSONL = os.path.join(
    DATA_DIR,
    "..",
    "..",
    "finetune",
    "training_data",
    "train_test",
    "test.jsonl",
)
USERS_FILE = os.path.join(DATA_DIR, "users.json")
EVALUATIONS_FILE = os.path.join(DATA_DIR, "evaluations.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "user_progress.json")

CATEGORIES = [
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "ADVERSE_EFFECTS_TOLERABILITY",
    "EFFICACY_EXPECTATION_MISMATCH",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "OTHER_UNKNOWN",
]

CATEGORY_DEFINITIONS = {
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": (
        "Discontinuation driven by clinical judgment — the provider determines the GLP-1 RA "
        "is no longer medically appropriate. Includes: guideline-based deprescribing, "
        "contraindications (pancreatitis history, gastroparesis, pregnancy, cirrhosis/sarcopenia), "
        "drug-drug interactions, dose reduction for hypoglycemia risk, shifting to preferred agent "
        "(e.g., SGLT2i for renoprotection), pre-surgical holds, or patient clinical trajectory "
        "making ongoing GLP-1 therapy inadvisable."
    ),
    "ADVERSE_EFFECTS_TOLERABILITY": (
        "Discontinuation primarily due to side effects or intolerable adverse reactions — "
        "GI symptoms (nausea, vomiting, diarrhea, constipation, bloating), injection site "
        "reactions, fatigue/dizziness/orthostasis attributed to the medication, gallbladder "
        "disease, appetite suppression beyond therapeutic intent, or any adverse effect leading "
        "to stop/dose-reduce."
    ),
    "EFFICACY_EXPECTATION_MISMATCH": (
        "Discontinuation or clinical concern because the medication is perceived or observed "
        "as no longer providing adequate benefit — weight loss plateau, A1c not improving or "
        "worsening despite adherence, patient perception that 'it stopped working,' waning "
        "response, or mismatch between expected and actual outcomes."
    ),
    "ACCESS_COST_SYSTEM_BARRIERS": (
        "Discontinuation due to external/systemic barriers — insurance denial, prior "
        "authorization failure, formulary restrictions, medication shortage/backorder, "
        "cost/affordability, pharmacy supply issues, or VA/non-VA coverage changes "
        "preventing continued access."
    ),
    "PATIENT_PREFERENCE_BELIEFS": (
        "Discontinuation driven by patient choice not primarily due to side effects or "
        "medical appropriateness — social stigma, family influence, misinformation/fear "
        "from media, desire to not be medication-dependent, preference for 'natural' "
        "approaches, injection fatigue, or psychosocial distress about long-term pharmacotherapy."
    ),
    "PLANNED_GOAL_BASED_DISCONTINUATION": (
        "Discontinuation that was anticipated from the start — time-limited trial with "
        "pre-defined endpoints met, goal weight/A1c achieved and maintained, planned "
        "taper/step-off after reaching targets, pre-conception/pregnancy planning, or "
        "structured deprescribing protocol where the original treatment plan included stopping."
    ),
    "OTHER_UNKNOWN": (
        "Discontinuation where the reason cannot be determined from the clinical note — "
        "unclear documentation, patient unable to explain, ambiguous chart entries, "
        "conflicting information, or insufficient detail to assign any of the above categories."
    ),
}

CONFIDENCE_LEVELS = ["High", "Medium", "Low"]


def load_json(filepath, default):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return default


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_notes():
    notes = []
    with open(TEST_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                notes.append(json.loads(line))
    return notes


NOTES = load_notes()
TOTAL_NOTES = len(NOTES)


def format_category_label(cat):
    pretty = cat.replace("_", " ").title()
    definition = CATEGORY_DEFINITIONS[cat]
    short = definition.split("—")[0].strip() if "—" in definition else definition.split(".")[0].strip()
    return f"**{pretty}** — {short}"


def build_instructions_md():
    lines = [
        "# Evaluation Instructions",
        "",
        "## Task",
        "You are evaluating synthetic clinical notes to verify that each note's **primary reason** "
        "for GLP-1 RA discontinuation or change matches the intended category.",
        "",
        "## Steps",
        "1. Read the full clinical note carefully.",
        "2. Determine the **primary reason** for GLP-1 RA discontinuation or change.",
        "3. Select the category that best represents the primary reason.",
        "4. Rate your confidence in the assignment (High / Medium / Low).",
        "5. If multiple reasons are present, select the **dominant/primary** driver.",
        "6. Add optional comments for ambiguous or difficult cases.",
        "7. Click **Submit** to save and advance to the next note.",
        "",
        "## Confidence Scale",
        "- **High**: The note clearly and unambiguously fits the selected category.",
        "- **Medium**: The note mostly fits the selected category but has some ambiguity.",
        "- **Low**: The note is ambiguous or could reasonably fit multiple categories.",
        "",
        "## Category Definitions",
        "",
    ]
    for cat in CATEGORIES:
        pretty = cat.replace("_", " ").title()
        definition = CATEGORY_DEFINITIONS[cat]
        lines.append(f"### {pretty}")
        lines.append("")
        lines.append(definition)
        lines.append("")
    return "\n".join(lines)


INSTRUCTIONS_MD = build_instructions_md()


class SessionState:
    def __init__(self):
        self.username = None
        self.current_index = 0


session = SessionState()


def login(username):
    username = username.strip()
    if not username:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Please enter a username.",
            "",
            None,
            None,
            None,
            "",
            0,
            "",
        )
    if not re.match(r"^[a-zA-Z0-9_\- ]{2,50}$", username):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Invalid username. Use letters, numbers, spaces, hyphens, or underscores (2-50 chars).",
            "",
            None,
            None,
            None,
            "",
            0,
            "",
        )

    users = load_json(USERS_FILE, [])
    progress = load_json(PROGRESS_FILE, {})

    if username not in users:
        users.append(username)
        save_json(USERS_FILE, users)

    if username not in progress:
        progress[username] = {
            "current_index": 0,
            "completed_indices": [],
            "last_active": datetime.now().isoformat(),
        }
        save_json(PROGRESS_FILE, progress)
    else:
        progress[username]["last_active"] = datetime.now().isoformat()
        save_json(PROGRESS_FILE, progress)

    session.username = username
    completed = progress[username]["completed_indices"]
    current_index = progress[username]["current_index"]

    if current_index >= TOTAL_NOTES:
        current_index = find_next_uncompleted(current_index, completed)

    if current_index >= TOTAL_NOTES:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"Welcome back, **{username}**! You have completed all {TOTAL_NOTES} evaluations. Thank you!",
            "",
            None,
            None,
            None,
            "",
            TOTAL_NOTES,
            "",
        )

    session.current_index = current_index
    note_text, note_id, progress_text, progress_bar = get_note_display(current_index, completed)

    return (
        gr.update(visible=True),
        gr.update(visible=False),
        f"Welcome, **{username}**! {progress_text}",
        note_text,
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        "",
        progress_bar,
        "",
    )


def find_next_uncompleted(start_index, completed_indices):
    completed_set = set(completed_indices)
    for i in range(start_index, TOTAL_NOTES):
        if i not in completed_set:
            return i
    for i in range(0, start_index):
        if i not in completed_set:
            return i
    return TOTAL_NOTES


def get_note_display(index, completed_indices):
    note = NOTES[index]
    note_id = note.get("note_id", f"note_{index}")
    clinical_note = note.get("clinical_note", "")
    completed_count = len(completed_indices)
    progress_text = f"You have completed **{completed_count}** of **{TOTAL_NOTES}** notes."
    progress_bar = completed_count
    return clinical_note, note_id, progress_text, progress_bar


def submit_evaluation(category, confidence, comment):
    if session.username is None:
        return (
            "Please log in first.",
            "",
            None,
            None,
            None,
            "",
            0,
            "",
        )

    if category is None:
        note_text = NOTES[session.current_index].get("clinical_note", "") if session.current_index < TOTAL_NOTES else ""
        return (
            "⚠️ Please select a category before submitting.",
            note_text,
            gr.update(),
            gr.update(),
            gr.update(),
            comment,
            gr.update(),
            "",
        )

    if confidence is None:
        note_text = NOTES[session.current_index].get("clinical_note", "") if session.current_index < TOTAL_NOTES else ""
        return (
            "⚠️ Please select a confidence level before submitting.",
            note_text,
            gr.update(),
            gr.update(),
            gr.update(),
            comment,
            gr.update(),
            "",
        )

    progress = load_json(PROGRESS_FILE, {})
    evaluations = load_json(EVALUATIONS_FILE, [])
    user_progress = progress.get(session.username, {
        "current_index": 0,
        "completed_indices": [],
        "last_active": datetime.now().isoformat(),
    })

    idx = session.current_index
    note = NOTES[idx]
    note_id = note.get("note_id", f"note_{idx}")
    model_category = note.get("primary_reason", "UNKNOWN")

    evaluation = {
        "username": session.username,
        "note_id": note_id,
        "note_index": idx,
        "model_category": model_category,
        "evaluator_category": category,
        "confidence": confidence,
        "comment": comment.strip() if comment else "",
        "timestamp": datetime.now().isoformat(),
    }
    evaluations.append(evaluation)
    save_json(EVALUATIONS_FILE, evaluations)

    completed = user_progress["completed_indices"]
    if idx not in completed:
        completed.append(idx)
    user_progress["completed_indices"] = completed

    next_index = idx + 1
    next_index = find_next_uncompleted(next_index, completed)
    user_progress["current_index"] = next_index
    user_progress["last_active"] = datetime.now().isoformat()
    progress[session.username] = user_progress
    save_json(PROGRESS_FILE, progress)

    completed_count = len(completed)
    progress_bar = completed_count

    if next_index >= TOTAL_NOTES:
        session.current_index = TOTAL_NOTES
        return (
            f"✅ Evaluation saved! You have completed all **{TOTAL_NOTES}** evaluations. Thank you, **{session.username}**!",
            "All notes have been evaluated. Thank you for your work!",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            progress_bar,
            "",
        )

    session.current_index = next_index
    next_note_text, next_note_id, progress_text, _ = get_note_display(next_index, completed)

    return (
        f"✅ Evaluation saved for note **{note_id}**! {progress_text}",
        next_note_text,
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        "",
        progress_bar,
        "",
    )


def go_to_previous():
    progress = load_json(PROGRESS_FILE, {})
    user_progress = progress.get(session.username, {
        "current_index": 0,
        "completed_indices": [],
    })
    completed = user_progress.get("completed_indices", [])
    completed_set = set(completed)

    prev_index = session.current_index - 1
    while prev_index >= 0 and prev_index not in completed_set:
        prev_index -= 1

    if prev_index < 0:
        return (
            "No previously evaluated notes to go back to.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    session.current_index = prev_index
    note_text, note_id, progress_text, _ = get_note_display(prev_index, completed)
    evaluations = load_json(EVALUATIONS_FILE, [])
    prev_eval = None
    for ev in reversed(evaluations):
        if ev["username"] == session.username and ev["note_index"] == prev_index:
            prev_eval = ev
            break

    prev_cat = prev_eval["evaluator_category"] if prev_eval else None
    prev_conf = prev_eval["confidence"] if prev_eval else None
    prev_comment = prev_eval.get("comment", "") if prev_eval else ""

    return (
        f"Viewing previously evaluated note **{note_id}**. {progress_text}",
        note_text,
        gr.update(visible=True, value=prev_cat),
        gr.update(visible=True, value=prev_conf),
        gr.update(visible=True, value=prev_comment if prev_comment else None),
        prev_comment,
        gr.update(),
        "",
    )


def go_to_next_uncompleted():
    progress = load_json(PROGRESS_FILE, {})
    user_progress = progress.get(session.username, {
        "current_index": 0,
        "completed_indices": [],
    })
    completed = user_progress.get("completed_indices", [])

    next_index = find_next_uncompleted(session.current_index + 1, completed)
    if next_index >= TOTAL_NOTES:
        return (
            "No more uncompleted notes.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    session.current_index = next_index
    note_text, note_id, progress_text, _ = get_note_display(next_index, completed)

    return (
        f"Skipped to next uncompleted note **{note_id}**. {progress_text}",
        note_text,
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        gr.update(visible=True, value=None),
        "",
        gr.update(),
        "",
    )


CSS = """
    .note-textbox textarea {
        font-family: monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    .category-radio label {
        font-weight: 500;
        padding: 4px 8px;
    }
    .progress-text {
        font-size: 16px;
        font-weight: 600;
    }
    #login-btn {
        min-width: 200px;
    }
"""

THEME = gr.themes.Soft()

with gr.Blocks(
    title="GLP-1 Primary Reason Category — Human Evaluation",
) as app:

    gr.Markdown(
        "# GLP-1 Primary Reason Category — Human Evaluation\n"
        "Evaluate synthetic clinical notes for correct primary reason category assignment."
    )

    with gr.Row():
        with gr.Column(scale=1):
            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter your username",
                max_lines=1,
            )
            login_btn = gr.Button("Login", variant="primary", elem_id="login-btn")
        with gr.Column(scale=2):
            login_message = gr.Markdown("")

    with gr.Tabs() as main_tabs:
        with gr.Tab("Evaluate"):
            eval_panel = gr.Column(visible=False)

            with eval_panel:
                status_text = gr.Markdown("", elem_classes=["progress-text"])

                with gr.Row():
                        progress_bar = gr.Slider(
                            minimum=0,
                            maximum=TOTAL_NOTES,
                            value=0,
                            step=1,
                            label="Progress",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=3):
                        note_display = gr.Textbox(
                            label="Clinical Note",
                            lines=25,
                            max_lines=50,
                            interactive=False,
                            elem_classes=["note-textbox"],
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### Select Primary Reason Category")
                        category_radio = gr.Radio(
                            choices=[(format_category_label(c), c) for c in CATEGORIES],
                            label="",
                            elem_classes=["category-radio"],
                            visible=False,
                        )

                        gr.Markdown("### Confidence Level")
                        confidence_radio = gr.Radio(
                            choices=CONFIDENCE_LEVELS,
                            label="",
                            visible=False,
                        )

                        comment_box = gr.Textbox(
                            label="Comments (optional)",
                            placeholder="Add any notes about this evaluation...",
                            lines=3,
                            visible=False,
                        )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", size="lg")
                prev_btn = gr.Button("← Previous", size="sm")
                skip_btn = gr.Button("Skip →", size="sm")

            submit_message = gr.Markdown("")

        status_md = gr.Markdown("", visible=False)

        with gr.Tab("Instructions & Categories"):
            gr.Markdown(INSTRUCTIONS_MD)

    login_btn.click(
        fn=login,
        inputs=[username_input],
        outputs=[
            eval_panel,
            login_btn,
            login_message,
            note_display,
            category_radio,
            confidence_radio,
            comment_box,
            status_md,
            progress_bar,
            submit_message,
        ],
    )

    username_input.submit(
        fn=login,
        inputs=[username_input],
        outputs=[
            eval_panel,
            login_btn,
            login_message,
            note_display,
            category_radio,
            confidence_radio,
            comment_box,
            status_md,
            progress_bar,
            submit_message,
        ],
    )

    submit_btn.click(
        fn=submit_evaluation,
        inputs=[category_radio, confidence_radio, comment_box],
        outputs=[
            submit_message,
            note_display,
            category_radio,
            confidence_radio,
            comment_box,
            status_md,
            progress_bar,
        ],
    )

    prev_btn.click(
        fn=go_to_previous,
        inputs=[],
        outputs=[
            submit_message,
            note_display,
            category_radio,
            confidence_radio,
            comment_box,
            status_md,
            progress_bar,
        ],
    )

    skip_btn.click(
        fn=go_to_next_uncompleted,
        inputs=[],
        outputs=[
            submit_message,
            note_display,
            category_radio,
            confidence_radio,
            comment_box,
            status_md,
            progress_bar,
        ],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=THEME,
        css=CSS,
    )
