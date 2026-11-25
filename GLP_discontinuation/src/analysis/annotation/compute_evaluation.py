import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from nltk.metrics.agreement import AnnotationTask

BASE_DIR = "/home/user/GLP_project/GLP_project_ad/code/annotator_evaluation"
OUTPUT_FILE = f"{BASE_DIR}/annotator_evaluation_results.md"

NORMALIZE_MAP = {"ADVERSE_EFFECTS_OR_TOLERABILITY": "ADVERSE_EFFECTS_TOLERABILITY"}

def normalize(label):
    return NORMALIZE_MAP.get(label, label)

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for ev in data["evaluations"].values():
        nid = ev["note_id"]
        out[nid] = {
            "assigned_label": normalize(ev["assigned_label"]),
            "human_corrected_label": normalize(ev["human_corrected_label"]),
        }
    return out

rater_1 = load_data(f"{BASE_DIR}/Rater_1.json")
rater_2 = load_data(f"{BASE_DIR}/Rater_2.json")
rater_3 = load_data(f"{BASE_DIR}/Rater_3.json")

all_note_ids = sorted(set(rater_1) & set(rater_2) & set(rater_3))
N = len(all_note_ids)

rater_1_h = [rater_1[nid]["human_corrected_label"] for nid in all_note_ids]
rater_2_h = [rater_2[nid]["human_corrected_label"] for nid in all_note_ids]
rater_3_h = [rater_3[nid]["human_corrected_label"] for nid in all_note_ids]
rater_1_m = [rater_1[nid]["assigned_label"] for nid in all_note_ids]
rater_2_m = [rater_2[nid]["assigned_label"] for nid in all_note_ids]
rater_3_m = [rater_3[nid]["assigned_label"] for nid in all_note_ids]

all_labels = sorted(set(rater_1_h + rater_2_h + rater_3_h + rater_1_m + rater_2_m + rater_3_m))

# ===== IAA =====
iaa_data = []
for nid in all_note_ids:
    iaa_data.append(("Rater_1", nid, rater_1[nid]["human_corrected_label"]))
    iaa_data.append(("Rater_2", nid, rater_2[nid]["human_corrected_label"]))
    iaa_data.append(("Rater_3", nid, rater_3[nid]["human_corrected_label"]))

task = AnnotationTask(data=iaa_data)
kappa_rater_1_rater_2 = task.kappa_pairwise("Rater_1", "Rater_2")
fleiss_kappa = task.multi_kappa()

# ===== Model vs Human =====
def compute_model_human(human_labels, model_labels, name):
    acc = accuracy_score(human_labels, model_labels)
    kappa = cohen_kappa_score(human_labels, model_labels, labels=all_labels)
    cm = confusion_matrix(human_labels, model_labels, labels=all_labels)
    return acc, kappa, cm

acc_rater_1, kappa_rater_1, cm_rater_1 = compute_model_human(rater_1_h, rater_1_m, "Rater_1")
acc_rater_2, kappa_rater_2, cm_rater_2 = compute_model_human(rater_2_h, rater_2_m, "Rater_2")
acc_rater_3, kappa_rater_3, cm_rater_3 = compute_model_human(rater_3_h, rater_3_m, "Rater_3")

# ===== Majority Vote Consensus =====
consensus_labels = []
consensus_details = []
for nid in all_note_ids:
    votes = [rater_1[nid]["human_corrected_label"], rater_2[nid]["human_corrected_label"], rater_3[nid]["human_corrected_label"]]
    c = Counter(votes)
    max_count = max(c.values())
    winners = [k for k, v in c.items() if v == max_count]
    if len(winners) == 1:
        consensus_labels.append(winners[0])
        consensus_details.append("unanimous" if max_count == 3 else "2-1 majority")
    else:
        consensus_labels.append(rater_3[nid]["human_corrected_label"])
        consensus_details.append("3-way tie -> Rater_3")

model_labels = [rater_1[nid]["assigned_label"] for nid in all_note_ids]
acc_consensus = accuracy_score(consensus_labels, model_labels)
kappa_consensus = cohen_kappa_score(consensus_labels, model_labels, labels=all_labels)
cm_consensus = confusion_matrix(consensus_labels, model_labels, labels=all_labels)

# ===== Error Analysis =====
def interpret_kappa(k):
    if k < 0: return "Poor (worse than chance)"
    if k < 0.20: return "Slight"
    if k < 0.40: return "Fair"
    if k < 0.60: return "Moderate"
    if k < 0.80: return "Substantial"
    return "Almost Perfect"

def cm_to_dict(cm, labels):
    d = {}
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            if cm[i, j] > 0:
                d[(true_l, pred_l)] = cm[i, j]
    return d

def top_confusions(cm, labels, top_k=10):
    confusions = []
    for i, true_l in enumerate(labels):
        for j, pred_l in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append((true_l, pred_l, int(cm[i, j])))
    confusions.sort(key=lambda x: -x[2])
    return confusions[:top_k]

confusions_rater_1 = top_confusions(cm_rater_1, all_labels)
confusions_rater_2 = top_confusions(cm_rater_2, all_labels)
confusions_rater_3 = top_confusions(cm_rater_3, all_labels)
confusions_consensus = top_confusions(cm_consensus, all_labels)

human_disagree_items = []
for idx, nid in enumerate(all_note_ids):
    labels_set = {rater_1_h[idx], rater_2_h[idx], rater_3_h[idx]}
    if len(labels_set) > 1:
        human_disagree_items.append({
            "note_id": nid,
            "rater_1": rater_1_h[idx],
            "rater_2": rater_2_h[idx],
            "rater_3": rater_3_h[idx],
            "consensus": consensus_labels[idx],
            "model": model_labels[idx],
            "detail": consensus_details[idx],
        })

pairwise_agreement_counts = {"Rater_1-Rater_2": 0, "Rater_1-Rater_3": 0, "Rater_2-Rater_3": 0}
for idx in range(N):
    if rater_1_h[idx] == rater_2_h[idx]:
        pairwise_agreement_counts["Rater_1-Rater_2"] += 1
    if rater_1_h[idx] == rater_3_h[idx]:
        pairwise_agreement_counts["Rater_1-Rater_3"] += 1
    if rater_2_h[idx] == rater_3_h[idx]:
        pairwise_agreement_counts["Rater_2-Rater_3"] += 1

# ===== Generate Markdown =====
def cm_table(cm, labels, title):
    lines = [f"**{title}**\n"]
    header = "| True \\ Pred | " + " | ".join(labels) + " |\n"
    sep = "|" + " --- |" * (len(labels) + 1) + "\n"
    lines.append(header)
    lines.append(sep)
    for i, true_l in enumerate(labels):
        row = f"| **{true_l}** |"
        for j in range(len(labels)):
            val = f" **{cm[i,j]}** " if i != j and cm[i,j] > 0 else f" {cm[i,j]} "
            row += val + "|"
        lines.append(row + "\n")
    return "".join(lines)

md = []
md.append("# Annotator Evaluation Results\n")
md.append(f"*Dataset:* {N} items annotated by 3 human evaluators (Rater_1, Rater_2, Rater_3)\n")
md.append(f"*Label categories:* {len(all_labels)}\n")
md.append(f"*Normalization:* `ADVERSE_EFFECTS_OR_TOLERABILITY` merged into `ADVERSE_EFFECTS_TOLERABILITY`\n")

md.append("## 1. Methodology & Metric Definitions\n")
md.append("### Cohen's Kappa ($\\kappa$)\n")
md.append("Measures inter-rater agreement for two raters, correcting for chance agreement:\n")
md.append("$$\\kappa = \\frac{p_o - p_e}{1 - p_e}$$\n")
md.append("where $p_o$ is observed agreement and $p_e$ is expected agreement by chance.\n")
md.append("Interpretation (Landis & Koch, 1977): $<0.20$ Slight, $0.20-0.40$ Fair, $0.40-0.60$ Moderate, $0.60-0.80$ Substantial, $0.80-1.00$ Almost Perfect.\n")

md.append("### Fleiss' Kappa ($\\kappa$)\n")
md.append("Generalization of Cohen's Kappa for more than two raters (Davies & Fleiss, 1982). ")
md.append("Computed as the average pairwise observed and expected agreement across all rater pairs.\n")

md.append("### Accuracy\n")
md.append("Simple proportion of exact label matches between model predictions and human labels:\n")
md.append("$$\\text{Accuracy} = \\frac{\\text{Number of correct predictions}}{\\text{Total number of predictions}}$$\n")

md.append("### Majority Vote Consensus (Gold Standard)\n")
md.append("For each item, the consensus label is determined by majority vote across the three annotators. ")
md.append("In case of a 3-way tie (all three annotators disagree), Rater_3's label is used as the tiebreaker.\n")

md.append("## 2. Inter-Annotator Agreement (IAA) Results\n")
md.append("### Pairwise Raw Agreement\n")
md.append("| Pair | Agreement Count | Agreement Rate |\n")
md.append("| --- | --- | --- |\n")
for pair, count in pairwise_agreement_counts.items():
    md.append(f"| {pair} | {count}/{N} | {count/N:.4f} |\n")
md.append("\n")

md.append("### Cohen's Kappa: Rater_1 vs Rater_2\n")
md.append(f"- **$\\kappa$ = {kappa_rater_1_rater_2:.4f}** ({interpret_kappa(kappa_rater_1_rater_2)})\n")
md.append(f"- This measures the agreement between Rater_1 and Rater_2 beyond what would be expected by chance.\n")

md.append("### Fleiss' Kappa: All Three Raters\n")
md.append(f"- **$\\kappa$ = {fleiss_kappa:.4f}** ({interpret_kappa(fleiss_kappa)})\n")
md.append(f"- This measures the global agreement across all three annotators simultaneously.\n")

md.append("### Human Disagreement Summary\n")
md.append(f"- Items where all 3 annotators agree: {N - len(human_disagree_items)}/{N} ({(N-len(human_disagree_items))/N:.1%})\n")
md.append(f"- Items with any disagreement: {len(human_disagree_items)}/{N} ({len(human_disagree_items)/N:.1%})\n")
three_way = [d for d in human_disagree_items if d["detail"] == "3-way tie -> Rater_3"]
two_one = [d for d in human_disagree_items if d["detail"] == "2-1 majority"]
md.append(f"  - 2-vs-1 majority disagreements: {len(two_one)}\n")
md.append(f"  - 3-way disagreements (all different): {len(three_way)}\n")

md.append("## 3. Model vs. Human Alignment Metrics\n")
md.append("### Per-Annotator Results\n")
md.append("| Annotator | Accuracy | Cohen's $\\kappa$ | Interpretation |\n")
md.append("| --- | --- | --- | --- |\n")
for name, acc, k in [("Rater_1", acc_rater_1, kappa_rater_1), ("Rater_2", acc_rater_2, kappa_rater_2), ("Rater_3", acc_rater_3, kappa_rater_3)]:
    md.append(f"| {name} | {acc:.4f} ({acc:.1%}) | {k:.4f} | {interpret_kappa(k)} |\n")
md.append("\n")

md.append("### Model vs. Consensus (Gold Standard)\n")
md.append(f"- **Accuracy:** {acc_consensus:.4f} ({acc_consensus:.1%})\n")
md.append(f"- **Cohen's $\\kappa$:** {kappa_consensus:.4f} ({interpret_kappa(kappa_consensus)})\n")

md.append("### Confusion Matrices\n\n")
md.append(cm_table(cm_rater_1, all_labels, "Model vs Rater_1"))
md.append("\n")
md.append(cm_table(cm_rater_2, all_labels, "Model vs Rater_2"))
md.append("\n")
md.append(cm_table(cm_rater_3, all_labels, "Model vs Rater_3"))
md.append("\n")
md.append(cm_table(cm_consensus, all_labels, "Model vs Consensus (Gold Standard)"))
md.append("\n")

md.append("## 4. Discussion & Error Analysis\n")

md.append("### Model Confusion Patterns (Top 10 per comparison)\n")
md.append("\n**Model vs Rater_1 — Most Common Misclassifications:**\n")
md.append("| True Label | Predicted Label | Count |\n")
md.append("| --- | --- | --- |\n")
for true_l, pred_l, cnt in confusions_rater_1:
    md.append(f"| {true_l} | {pred_l} | {cnt} |\n")
md.append("\n")

md.append("\n**Model vs Rater_2 — Most Common Misclassifications:**\n")
md.append("| True Label | Predicted Label | Count |\n")
md.append("| --- | --- | --- |\n")
for true_l, pred_l, cnt in confusions_rater_2:
    md.append(f"| {true_l} | {pred_l} | {cnt} |\n")
md.append("\n")

md.append("\n**Model vs Rater_3 — Most Common Misclassifications:**\n")
md.append("| True Label | Predicted Label | Count |\n")
md.append("| --- | --- | --- |\n")
for true_l, pred_l, cnt in confusions_rater_3:
    md.append(f"| {true_l} | {pred_l} | {cnt} |\n")
md.append("\n")

md.append("\n**Model vs Consensus — Most Common Misclassifications:**\n")
md.append("| True Label | Predicted Label | Count |\n")
md.append("| --- | --- | --- |\n")
for true_l, pred_l, cnt in confusions_consensus:
    md.append(f"| {true_l} | {pred_l} | {cnt} |\n")
md.append("\n")

md.append("### Key Observations\n")

model_correct_count = sum(1 for i in range(N) if model_labels[i] == consensus_labels[i])
model_wrong_count = N - model_correct_count
md.append(f"1. **Overall Model Performance:** The model correctly matches the consensus gold standard on {model_correct_count}/{N} items ({acc_consensus:.1%}). ")
md.append(f"The Cohen's $\\kappa$ of {kappa_consensus:.4f} ({interpret_kappa(kappa_consensus)}) indicates the model's agreement beyond chance.\n")

md.append(f"2. **Human Agreement Level:** The Fleiss' $\\kappa$ of {fleiss_kappa:.4f} ({interpret_kappa(fleiss_kappa)}) across all three annotators suggests ")
if fleiss_kappa >= 0.80:
    md.append("almost perfect agreement, indicating a well-defined annotation schema.\n")
elif fleiss_kappa >= 0.60:
    md.append("substantial agreement, indicating a reasonably clear annotation schema with some edge cases.\n")
elif fleiss_kappa >= 0.40:
    md.append("moderate agreement, suggesting notable subjectivity in the annotation schema.\n")
else:
    md.append("only fair/slight agreement, indicating significant ambiguity in the annotation guidelines.\n")

md.append(f"3. **Rater_1 vs Rater_2 Agreement:** Cohen's $\\kappa$ = {kappa_rater_1_rater_2:.4f} ({interpret_kappa(kappa_rater_1_rater_2)}). ")
md.append(f"Raw agreement rate: {pairwise_agreement_counts['Rater_1-Rater_2']}/{N} ({pairwise_agreement_counts['Rater_1-Rater_2']/N:.1%}).\n")

md.append(f"4. **Label Distribution Skew:** The most frequent label is `{Counter(consensus_labels).most_common(1)[0][0]}` ")
md.append(f"({Counter(consensus_labels).most_common(1)[0][1]} items, {Counter(consensus_labels).most_common(1)[0][1]/N:.1%}). ")
md.append("This class imbalance may inflate accuracy scores; Cohen's $\\kappa$ provides a more reliable measure by correcting for chance.\n")

md.append(f"5. **Areas of Human Disagreement:** {len(human_disagree_items)} items ({len(human_disagree_items)/N:.1%}) had at least one disagreement among the three annotators. ")
if len(three_way) > 0:
    md.append(f"Of these, {len(three_way)} items had all three annotators assigning different labels, indicating genuinely ambiguous cases.\n")
else:
    md.append("All disagreements were 2-vs-1 splits, suggesting most items have clear majority interpretation.\n")

md.append("6. **Model Error Patterns:** The most common model misclassifications tend to involve:\n")
if confusions_consensus:
    top3 = confusions_consensus[:3]
    for true_l, pred_l, cnt in top3:
        md.append(f"   - True `{true_l}` → Predicted `{pred_l}` ({cnt} instances)\n")

md.append(f"\n7. **Model vs Individual Annotators:** The model's $\\kappa$ scores range from {min(kappa_rater_1, kappa_rater_2, kappa_rater_3):.4f} to {max(kappa_rater_1, kappa_rater_2, kappa_rater_3):.4f} across individual annotators. ")
best_annotator = max([("Rater_1", kappa_rater_1), ("Rater_2", kappa_rater_2), ("Rater_3", kappa_rater_3)], key=lambda x: x[1])
md.append(f"The model aligns best with {best_annotator[0]} ($\\kappa$ = {best_annotator[1]:.4f}).\n")

md.append("### Recommendations\n")
md.append("- **Annotation Guidelines:** Review the label pairs that show the most human disagreement to refine annotation guidelines.\n")
md.append("- **Model Improvement:** Focus model training on the most common confusion pairs identified above.\n")
md.append("- **Ambiguous Cases:** Items with 3-way human disagreement may benefit from expert review or additional context.\n")

with open(OUTPUT_FILE, "w") as f:
    f.write("".join(md))

print(f"Report written to {OUTPUT_FILE}")
print(f"N={N}, Labels={len(all_labels)}")
print(f"Cohen's Kappa (Rater_1-Rater_2): {kappa_rater_1_rater_2:.4f}")
print(f"Fleiss' Kappa (All 3): {fleiss_kappa:.4f}")
print(f"Model vs Rater_1: Acc={acc_rater_1:.4f}, Kappa={kappa_rater_1:.4f}")
print(f"Model vs Rater_2: Acc={acc_rater_2:.4f}, Kappa={kappa_rater_2:.4f}")
print(f"Model vs Rater_3: Acc={acc_rater_3:.4f}, Kappa={kappa_rater_3:.4f}")
print(f"Model vs Consensus: Acc={acc_consensus:.4f}, Kappa={kappa_consensus:.4f}")
