#!/usr/bin/env python3
"""Generate the final label_independence_report.md.

Compiles results from:
  - existing_annotation_metrics.json  (baseline)
  - human_label_independence_metrics.json  (blind annotation)
  - llm_blind_judge_metrics.json  (LLM re-judging)
  - metadata_audit_report.md  (metadata audit)

Produces a structured report for the rebuttal.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def cm_to_markdown_table(cm, labels, row_label="True", col_label="Pred", highlight_diag=True):
    """Convert confusion matrix to markdown table string."""
    lines = []
    header = f"| {row_label} \\ {col_label} | " + " | ".join(labels) + " |\n"
    sep = "|" + " --- |" * (len(labels) + 1) + "\n"
    lines.append(header)
    lines.append(sep)
    for i, true_l in enumerate(labels):
        row = f"| **{true_l}** |"
        for j in range(len(labels)):
            val = cm[i][j]
            if i == j and highlight_diag:
                row += f" **{val}** |"
            else:
                row += f" {val} |"
        lines.append(row + "\n")
    return "".join(lines)


def format_per_class_table(per_class, title="Per-Class Metrics"):
    lines = [f"### {title}\n"]
    lines.append("| Category | Precision | Recall | F1 | Support |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for cat in CATEGORIES:
        if cat in per_class:
            pc = per_class[cat]
            lines.append(
                f"| {cat} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                f"{pc['f1']:.4f} | {pc['support']} |\n"
            )
    return "".join(lines)


def main():
    existing = load_json(RESULTS_DIR / "existing_annotation_metrics.json")
    blind = load_json(RESULTS_DIR / "human_label_independence_metrics.json")
    llm = load_json(RESULTS_DIR / "llm_blind_judge_metrics.json")

    report = []
    report.append("# Label-Independence Validation Report\n")
    report.append("\n---\n")
    report.append("## Experiment Overview\n")
    report.append("\n")
    report.append("This report presents three experiments assessing whether the primary discontinuation "
                  "reason can be inferred from the clinical note text alone, without exposing annotators "
                  "or models to the generation target label, generator metadata, file paths, model "
                  "predictions, or original LLM-judge scores.\n")
    report.append("\n")
    report.append("### Experiments\n")
    report.append("1. **Baseline (label-conditioned)**: Re-analysis of existing annotation where annotators "
                  "saw a pre-filled label and verified it.\n")
    report.append("2. **Label-blind human re-annotation**: Three annotators assigned labels from scratch "
                  "seeing only note text and category definitions.\n")
    report.append("3. **Label-blind LLM re-judging**: `google/gemma-4-31B-it` classified the same blinded "
                  "notes via a locally hosted vLLM endpoint, without label exposure.\n")
    report.append("\n---\n")

    # ============================
    # Section 1: Baseline
    # ============================
    report.append("## 1. Baseline: Existing Label-Conditioned Annotation\n")
    report.append("\n")
    if existing:
        report.append(f"**Method:** The original annotation presented annotators with a pre-populated "
                      f"label (`assigned_label`) and asked whether they agreed. The `human_corrected_label` "
                      f"field records their final judgment. This is a label-conditioned annotation, not blind.\n")
        report.append("\n")
        iaa = existing.get("inter_annotator_agreement", existing)
        fleiss = iaa.get("fleiss_kappa", existing.get("fleiss_kappa", "N/A"))
        fleiss_interp = iaa.get("fleiss_kappa_interpretation", existing.get("fleiss_kappa_interpretation", ""))
        unanimous = iaa.get("unanimous_agreement", existing.get("unanimous_agreement", {}))
        pairwise = iaa.get("pairwise_agreement", existing.get("pairwise_agreement", {}))
        pairwise_kappa = iaa.get("pairwise_cohens_kappa", existing.get("pairwise_cohens_kappa", {}))

        report.append("### Inter-Annotator Agreement\n")
        report.append(f"- **Fleiss' kappa**: {fleiss} ({fleiss_interp})\n")
        if isinstance(unanimous, dict):
            report.append(f"- **Unanimous agreement**: {unanimous.get('count', 'N/A')}/"
                          f"{unanimous.get('total', 'N/A')} "
                          f"({unanimous.get('percentage', 0):.1f}%)\n")
        else:
            report.append(f"- **Unanimous agreement**: {unanimous}\n")

        report.append("\n#### Pairwise Agreement\n")
        report.append("| Pair | Agreement Rate | Cohen's κ |\n")
        report.append("| --- | --- | --- |\n")
        pairs_display = {
            "Anu-Parth": ("Anu", "Parth"),
            "Anu-adiba": ("Anu", "adiba"),
            "Parth-adiba": ("Parth", "adiba"),
            "Anu-Parth": ("Anu", "Parth"),
        }
        # Handle different key formats
        for pair_key in pairwise:
            parts = pair_key.split("-")
            if len(parts) == 2:
                rate = pairwise[pair_key].get("rate", pairwise[pair_key])
                k = pairwise_kappa.get(pair_key, "N/A")
                if isinstance(k, float):
                    report.append(f"| {pair_key} | {rate:.4f} | {k:.4f} |\n")
                else:
                    report.append(f"| {pair_key} | {rate:.4f} | N/A |\n")

        consensus_data = existing.get("consensus_vs_assigned_label",
                                       existing.get("consensus_vs_assigned_label", {}))
        if consensus_data:
            report.append("\n### Consensus vs Intended Label\n")
            report.append(f"- **Accuracy**: {consensus_data.get('accuracy', 'N/A')}\n")
            report.append(f"- **Cohen's κ**: {consensus_data.get('cohens_kappa', 'N/A')} "
                          f"({consensus_data.get('interpretation', '')})\n")

        report.append("\n**Caveat:** This baseline reflects annotation where the intended label was visible. "
                      "It overestimates label recoverability from text alone.\n")
    else:
        report.append("Baseline results not available. Run `compute_existing_annotation.py` first.\n")
    report.append("\n---\n")

    # ============================
    # Section 2: Blind Human Annotation
    # ============================
    report.append("## 2. Label-Blind Human Re-Annotation\n")
    report.append("\n")
    if blind:
        report.append("**Method:** Three annotators (Anu, Parth, Adiba) independently re-annotated "
                      "the 320-note test set from scratch. Each annotator saw ONLY the clinical note "
                      "text and the seven category definitions. The intended label, model predictions, "
                      "generation branch, category folder, and all metadata were hidden. Annotators "
                      "worked independently without discussion. After independent annotation, a consensus "
                      "label was assigned by majority vote. Cases where all three annotators disagreed "
                      "were flagged as adjudicated.\n")
        report.append("\n")

        iaa = blind.get("inter_annotator_agreement", {})
        fleiss = iaa.get("fleiss_kappa", "N/A")
        fleiss_interp = iaa.get("fleiss_kappa_interpretation", "")
        unanimous = iaa.get("unanimous_agreement", {})
        disagreement = iaa.get("disagreement_summary", {})
        pairwise = iaa.get("pairwise_agreement", {})
        pairwise_kappa = iaa.get("pairwise_cohens_kappa", {})

        report.append("### Inter-Annotator Agreement\n")
        report.append(f"- **Fleiss' kappa**: {fleiss} ({fleiss_interp})\n")
        report.append(f"- **Unanimous agreement**: {unanimous.get('count', 'N/A')}/"
                      f"{unanimous.get('total', 'N/A')} "
                      f"({unanimous.get('percentage', 0):.1f}%)\n")
        report.append(f"- **Any disagreement**: {disagreement.get('any_disagreement_count', 'N/A')}/"
                      f"{disagreement.get('percentage', 0):.1f}%\n")
        report.append(f"- **2-vs-1 majority splits**: {disagreement.get('two_vs_one', 'N/A')}\n")
        report.append(f"- **Adjudicated (3-way)**: {disagreement.get('adjudicated_three_way', 'N/A')}\n")
        report.append("\n")

        report.append("#### Pairwise Agreement\n")
        report.append("| Pair | Agreement Rate | Cohen's κ |\n")
        report.append("| --- | --- | --- |\n")
        for pair_key in pairwise:
            rate = pairwise[pair_key].get("rate", 0)
            k = pairwise_kappa.get(pair_key, "N/A")
            if isinstance(k, float):
                report.append(f"| {pair_key} | {rate:.4f} | {k:.4f} |\n")
            else:
                report.append(f"| {pair_key} | {rate:.4f} | N/A |\n")

        # Consensus vs intended
        cvi = blind.get("consensus_vs_intended", {})
        report.append("\n### Blind Consensus vs Intended Labels\n")
        if cvi:
            report.append(f"- **Accuracy**: {cvi.get('accuracy', 'N/A'):.4f}\n")
            report.append(f"- **Cohen's κ**: {cvi.get('cohens_kappa', 'N/A'):.4f} "
                          f"({cvi.get('kappa_interpretation', '')})\n")
            report.append(f"- **Macro F1**: {cvi.get('macro_f1', 'N/A'):.4f}\n")
            report.append(f"- **Weighted F1**: {cvi.get('weighted_f1', 'N/A'):.4f}\n")
            report.append("\n")

            # Per-class table
            per_class = cvi.get("per_class", {})
            report.append(format_per_class_table(per_class, "Per-Class Metrics (Blind Consensus vs Intended)"))
            report.append("\n")

            # Confusion matrix
            cm = cvi.get("confusion_matrix", [])
            if cm:
                report.append("### Confusion Matrix: Intended × Blind Consensus\n")
                report.append(cm_to_markdown_table(cm, CATEGORIES, "Intended", "Blind Consensus"))
                report.append("\n")

        # Top confusions
        top_conf = blind.get("top_confusions_vs_intended", [])
        if top_conf:
            report.append("### Most Common Confusions (Consensus vs Intended)\n")
            report.append("| Intended Label | Consensus Label | Count |\n")
            report.append("| --- | --- | --- |\n")
            for tc in top_conf[:10]:
                report.append(f"| {tc['intended']} | {tc['predicted']} | {tc['count']} |\n")
            report.append("\n")

        # Confidence
        conf_annot = blind.get("confidence_by_annotator", {})
        if conf_annot:
            report.append("### Annotator Confidence\n")
            report.append("| Annotator | Mean Confidence | Median | Std |\n")
            report.append("| --- | --- | --- | --- |\n")
            for name in ["Anu", "Parth", "Adiba"]:
                if name in conf_annot:
                    c = conf_annot[name]
                    report.append(f"| {name} | {c['mean']:.3f} | {c['median']:.3f} | {c['std']:.3f} |\n")
            report.append("\n")

        # Label distributions
        dists = blind.get("label_distributions", {})
        if dists:
            report.append("### Label Distribution Comparison\n")
            report.append("| Category | Intended | Blind Consensus |\n")
            report.append("| --- | --- | --- |\n")
            intended_dist = dists.get("intended", {})
            consensus_dist = dists.get("blind_consensus", {})
            for cat in CATEGORIES:
                iv = intended_dist.get(cat, 0)
                cv = consensus_dist.get(cat, 0)
                report.append(f"| {cat} | {iv} | {cv} |\n")
            report.append("\n")

        report.append("### Comparison with Baseline\n")
        if existing:
            existing_fleiss = existing.get("inter_annotator_agreement", existing).get(
                "fleiss_kappa", existing.get("fleiss_kappa", "N/A"))
            report.append(f"- Baseline (label-conditioned) Fleiss' κ: {existing_fleiss}\n")
            report.append(f"- Blind Fleiss' κ: {fleiss}\n")
            report.append(f"- Difference: The blind annotation removes the priming effect of seeing "
                          f"a pre-filled label. If blind agreement is comparable to the baseline, "
                          f"this suggests labels are largely recoverable from text alone.\n")
    else:
        report.append("Blind annotation results not available. "
                      "Run the annotation UI and `compute_human_metrics.py` first.\n")
    report.append("\n---\n")

    # ============================
    # Section 3: LLM Blind Judging
    # ============================
    report.append("## 3. Label-Blind LLM Re-Judging\n")
    report.append("\n")
    if llm:
        report.append("**Method:** We ran `google/gemma-4-31B-it` (hosted on vLLM at "
                      "`http://localhost:8091/v1`, GPU 4 / H200) on the same 320 blinded notes. "
                      "The model received only the task description, seven category definitions, "
                      "and the clinical note text. It was asked to return a JSON object with "
                      "`primary_reason`, `confidence`, `supporting_evidence`, `secondary_factors`, "
                      "and `ambiguity_note`. No intended label or generation metadata was provided.\n")
        report.append("\n")

        # LLM vs Intended
        if "llm_vs_intended_labels" in llm:
            m = llm["llm_vs_intended_labels"]
            report.append("### LLM vs Intended Labels\n")
            report.append(f"- **Accuracy**: {m.get('accuracy', 'N/A'):.4f}\n")
            report.append(f"- **Cohen's κ**: {m.get('cohens_kappa', 'N/A'):.4f} "
                          f"({m.get('kappa_interpretation', '')})\n")
            report.append(f"- **Macro F1**: {m.get('macro_f1', 'N/A'):.4f}\n")
            report.append(f"- **Weighted F1**: {m.get('weighted_f1', 'N/A'):.4f}\n")
            report.append(f"- **Valid predictions**: {m.get('valid_samples', 'N/A')}/"
                          f"{m.get('total_samples', 'N/A')}\n")
            report.append("\n")

            per_class = m.get("per_class", {})
            report.append(format_per_class_table(per_class, "LLM Per-Class Metrics (vs Intended)"))
            report.append("\n")

            cm = m.get("confusion_matrix", [])
            if cm:
                report.append("#### Confusion Matrix: Intended × LLM\n")
                report.append(cm_to_markdown_table(cm, CATEGORIES, "Intended", "LLM Prediction"))
                report.append("\n")

        # LLM vs Human Consensus
        if "llm_vs_human_consensus" in llm:
            m = llm["llm_vs_human_consensus"]
            report.append("### LLM vs Human Consensus\n")
            report.append(f"- **Accuracy**: {m.get('accuracy', 'N/A'):.4f}\n")
            report.append(f"- **Cohen's κ**: {m.get('cohens_kappa', 'N/A'):.4f} "
                          f"({m.get('kappa_interpretation', '')})\n")
            report.append(f"- **Macro F1**: {m.get('macro_f1', 'N/A'):.4f}\n")
            report.append(f"- **Weighted F1**: {m.get('weighted_f1', 'N/A'):.4f}\n")
            report.append("\n")

            cm = m.get("confusion_matrix", [])
            if cm:
                report.append("#### Confusion Matrix: Human Consensus × LLM\n")
                report.append(cm_to_markdown_table(cm, CATEGORIES, "Human Consensus", "LLM Prediction"))
                report.append("\n")

        # LLM + Human agree against intended
        if "llm_human_agree_against_intended" in llm:
            m = llm["llm_human_agree_against_intended"]
            report.append("### Cases Where LLM and Humans Agree Against Intended Label\n")
            report.append(f"- **Count**: {m.get('count', 0)}/{m.get('total_joint', 0)} "
                          f"({m.get('percentage', 0):.1f}%)\n")
            cases = m.get("cases", [])
            if cases:
                report.append("| Blind ID | Intended | Human Consensus | LLM | LLM Confidence |\n")
                report.append("| --- | --- | --- | --- | --- |\n")
                for c in cases[:15]:
                    report.append(f"| {c.get('blind_id', '')} | {c.get('intended_label', '')} | "
                                  f"{c.get('human_consensus', '')} | {c.get('llm_prediction', '')} | "
                                  f"{c.get('llm_confidence', 0):.2f} |\n")
                report.append("\n")

        # Low-confidence predictions
        if "low_confidence_predictions" in llm:
            m = llm["low_confidence_predictions"]
            report.append("### Low-Confidence Predictions\n")
            report.append(f"- **Threshold**: < {m.get('threshold', 0.6)}\n")
            report.append(f"- **Count**: {m.get('count', 0)}/{m.get('total', 0)} "
                          f"({m.get('percentage', 0):.1f}%)\n")
            cases = m.get("cases", [])
            if cases:
                report.append("| Blind ID | Predicted | Confidence | Intended | Ambiguity Note |\n")
                report.append("| --- | --- | --- | --- | --- |\n")
                for c in cases[:10]:
                    report.append(f"| {c.get('blind_id', '')} | {c.get('primary_reason', '')} | "
                                  f"{c.get('confidence', 0):.2f} | {c.get('intended_label', '')} | "
                                  f"{c.get('ambiguity_note', '')[:80]} |\n")
                report.append("\n")

        # Errors
        if "parse_and_api_errors" in llm:
            m = llm["parse_and_api_errors"]
            report.append(f"### Parse and API Errors\n")
            report.append(f"- **Count**: {m.get('count', 0)}/{m.get('total', 0)}\n")
            report.append("\n")
    else:
        report.append("LLM results not available. Run `run_llm_blind_judging.py` and "
                      "`compute_llm_metrics.py` first.\n")
    report.append("\n---\n")

    # ============================
    # Section 4: Metadata Audit
    # ============================
    report.append("## 4. Metadata Removal Audit\n")
    report.append("\n")
    audit_path = RESULTS_DIR / "metadata_audit_report.md"
    if audit_path.exists():
        report.append("See [metadata_audit_report.md](../results/metadata_audit_report.md) for full details.\n")
        report.append("\n")
        # Include a summary section
        report.append("### Audit Summary\n")
        # Count lines from audit report
        with open(audit_path) as f:
            audit_content = f.read()
        # Extract the summary section
        summary_start = audit_content.find("## Summary")
        if summary_start >= 0:
            summary = audit_content[summary_start:]
            report.append(summary)
    else:
        report.append("Metadata audit not available. Run `metadata_removal_audit.py` first.\n")
    report.append("\n---\n")

    # ============================
    # Section 5: Discussion
    # ============================
    report.append("## 5. Discussion\n")
    report.append("\n")

    # Build discussion based on available data
    if blind:
        unanimous_pct = blind.get("inter_annotator_agreement", {}).get(
            "unanimous_agreement", {}).get("percentage", 0)
        fleiss_b = blind.get("inter_annotator_agreement", {}).get("fleiss_kappa", 0)
        acc_cvi = blind.get("consensus_vs_intended", {}).get("accuracy", 0)

        if unanimous_pct > 70 and fleiss_b > 0.7:
            report.append("**Interpretation:** The label-blind annotation produced substantial-to-almost-perfect "
                          "agreement among annotators. This suggests that the primary discontinuation reason "
                          "is largely recoverable from the clinical note text alone, even without exposure "
                          "to the generation target label. The majority-consensus labels align well with "
                          "the intended labels, supporting label coherence under the controlled taxonomy.\n")
        elif unanimous_pct > 50:
            report.append("**Interpretation:** The label-blind annotation produced moderate agreement. "
                          "While many labels are recoverable from text, meaningful ambiguity remains. "
                          "This suggests the dataset should be framed as a controlled primary-reason "
                          "benchmark rather than a natural estimate of real-world label agreement.\n")
        else:
            report.append("**Interpretation:** The label-blind annotation revealed substantial label "
                          "ambiguity when the intended label is not provided. This confirms the reviewer's "
                          "concern that labels are in part guaranteed by construction. We will reframe "
                          "the benchmark accordingly.\n")

    if blind and existing:
        fleiss_e = existing.get("inter_annotator_agreement", existing).get("fleiss_kappa",
                         existing.get("fleiss_kappa", 0))
        report.append(f"\n**Baseline vs blind comparison:** The label-conditioned baseline "
                      f"(Fleiss' κ = {fleiss_e}) and the label-blind annotation "
                      f"(Fleiss' κ = {fleiss_b}) show the effect of removing the pre-filled label."
                      f" This quantifies how much agreement is driven by the label itself vs. "
                      f"recoverable from text.\n")

    report.append("\n### Representative Ambiguous Cases\n")
    report.append("\n")
    report.append("Cases where annotators disagreed or where blind consensus differed from the "
                  "intended label illustrate the boundary of the taxonomy. Common patterns include:\n")
    report.append("\n")
    report.append("- **Adverse effects vs. clinical decision**: A note may describe both nausea "
                  "(adverse effect) and a clinician's decision to stop due to renal function decline "
                  "(clinical decision). The dominant reason may be interpreted differently.\n")
    report.append("- **Patient preference vs. planned goal**: A patient who achieved weight loss and "
                  "chooses to stop could be classified under either category depending on whether the "
                  "clinician's plan or the patient's initiative is viewed as primary.\n")
    report.append("- **Efficacy mismatch vs. other/unknown**: When the note is ambiguous about "
                  "why the medication stopped, annotators may differ on whether the text supports "
                  "a specific label or belongs in the catch-all category.\n")
    report.append("\n")
    report.append("*Detailed disagreement cases will be added after annotation review.*\n")
    report.append("\n")

    # ============================
    # Section 6: Limitations
    # ============================
    report.append("## 6. Limitations\n")
    report.append("\n")
    report.append("Even if blind agreement is high, the following limitations remain:\n")
    report.append("\n")
    report.append("1. **Target-label-conditioned generation:** All notes were generated from target "
                  "labels. This experiment supports label coherence and text-level recoverability, "
                  "but it does not fully reproduce naturally occurring label uncertainty in real "
                  "clinical documentation.\n")
    report.append("2. **Synthetic note artifacts:** The notes were generated by language models and "
                  "may contain rhetorical patterns, section structures, or clinical framing that make "
                  "the primary reason more salient than in real notes.\n")
    report.append("3. **Controlled taxonomy:** The seven categories are predefined and annotators are "
                  "forced to choose one. Real clinical notes may have genuinely multi-factorial "
                  "discontinuations that do not fit neatly into a single category.\n")
    report.append("4. **Annotator expertise:** All three annotators have biomedical research "
                  "experience and were familiar with the taxonomy from the baseline annotation. "
                  "This may not generalize to naive annotators or real clinical coders.\n")
    report.append("\n")

    # ============================
    # Section 7: Suggested Rebuttal Framing
    # ============================
    report.append("## 7. Suggested Rebuttal Framing\n")
    report.append("\n")

    if blind and unanimous_pct > 70:
        report.append("> To test whether labels were recoverable from the note text rather than only "
                      "guaranteed by construction, we performed a label-blind re-annotation of the final "
                      "320-note test set. Annotators saw only the note text and label definitions, with "
                      "intended labels, model predictions, generation branch, and category metadata hidden. "
                      f"The blinded annotation produced Fleiss' kappa {fleiss_b:.4f}, unanimous agreement "
                      f"on {blind.get('inter_annotator_agreement', {}).get('unanimous_agreement', {}).get('count', 0)}/320 notes, "
                      f"and majority-consensus agreement with the intended labels on "
                      f"{acc_cvi*320:.0f}/320 notes. "
                      "These results support label coherence under the controlled taxonomy, although they "
                      "do not eliminate the limitation that the synthetic notes were generated from target labels.\n")
    else:
        report.append("> The label-blind audit showed that many labels are recoverable from the note text, "
                      "but some classes remain ambiguous under independent review. We will report these "
                      "disagreements and revise the benchmark framing to emphasize that the dataset is a "
                      "controlled primary-reason benchmark rather than a natural estimate of real-world "
                      "label uncertainty.\n")

    report.append("\n---\n")
    report.append(f"*Report generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    out_path = REPORTS_DIR / "label_independence_report.md"
    with open(out_path, "w") as f:
        f.write("".join(report))
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
