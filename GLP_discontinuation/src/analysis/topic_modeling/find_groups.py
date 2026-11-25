import json
import re
import numpy as np
from tqdm import tqdm
import requests


class GroupLabeler:

    def __init__(self, topic_modeling_instance, vllm_url, model_name="google/gemma-4-31B-it"):
        self.tm = topic_modeling_instance
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.group_labels = {}

    def label_groups(self, save_dir, topic_labels_dict=None):
        if topic_labels_dict is None:
            with open(f'{save_dir}/topic_labels.json', 'r') as fh:
                topic_labels_dict = {int(k): v for k, v in json.load(fh).items()}

        topics_arr = np.array(self.tm.topics)
        valid_mask = topics_arr != -1
        valid_topics = topics_arr[valid_mask]
        unique_topics = sorted(np.unique(valid_topics))
        topic_to_idx = {t: i for i, t in enumerate(unique_topics)}

        groups = self.tm.groups
        unique_groups = sorted(np.unique(groups))

        print(f"Labeling {len(unique_groups)} groups via vLLM (chat completions)...")

        for group in tqdm(unique_groups):
            group_topic_indices = np.where(groups == group)[0]
            topic_labels_for_group = []
            for idx in group_topic_indices:
                original_topic = unique_topics[idx]
                label = topic_labels_dict.get(int(original_topic), "")
                if label:
                    topic_labels_for_group.append(label)

            combined_labels = "\n".join(topic_labels_for_group)

            messages = self._build_group_messages(group, combined_labels)
            response = self._query_vllm_chat(messages)
            self.group_labels[int(group)] = self._parse_response(response)

        with open(f'{save_dir}/group_labels.json', 'w') as fh:
            json.dump(self.group_labels, fh, indent=2)

        print(f"Saved group labels to {save_dir}/group_labels.json")
        return self.group_labels

    def _build_group_messages(self, group, topic_labels):
        system_prompt = (
            "You are a clinical informatics expert. "
            "Your task is to create ONE overarching label for a group of related clinical discussion topics about GLP-1 receptor agonist medications. "
            "Be specific and clinical. Output only the label, nothing else."
        )

        user_prompt = f"""The group contains the following topic labels:
{topic_labels}

Based on these topic labels, create ONE short, descriptive label (maximum 10 words) that captures the overarching theme of this group. Output only the label."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _query_vllm_chat(self, messages, max_tokens=100, temperature=0.1):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(f"{self.vllm_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _parse_response(self, text):
        text = text.strip().strip('"').strip("'")
        text = text.split("\n")[0].strip()
        return text
