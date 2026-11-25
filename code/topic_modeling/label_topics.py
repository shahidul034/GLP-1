import json
import os
import re
import pickle
import numpy as np
from tqdm import tqdm
import requests


class TopicLabeler:

    def __init__(self, topic_modeling_instance, vllm_url, model_name="google/gemma-4-31B-it"):
        self.tm = topic_modeling_instance
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.topic_labels = {}

    def label_topics(self, save_dir, n_reps=5):
        rep_docs = self.tm.find_representative_docs(n_reps=n_reps)
        rand_docs = self.tm.find_random_docs(n_reps=n_reps)

        topics_arr = np.array(self.tm.topics)
        valid_mask = topics_arr != -1
        valid_topics = topics_arr[valid_mask]
        unique_topics = sorted(np.unique(valid_topics))

        print(f"Labeling {len(unique_topics)} topics via vLLM (chat completions)...")

        for topic in tqdm(unique_topics):
            idx = list(unique_topics).index(topic)
            messages = self._build_messages(topic, rep_docs[idx], rand_docs[idx])
            response = self._query_vllm_chat(messages)
            self.topic_labels[int(topic)] = self._parse_response(response)

        with open(f'{save_dir}/topic_labels.json', 'w') as fh:
            json.dump(self.topic_labels, fh, indent=2)

        print(f"Saved topic labels to {save_dir}/topic_labels.json")
        return self.topic_labels

    def _build_messages(self, topic, rep_docs, rand_docs):
        keywords = self.tm.topic_model.get_topic(topic)
        keyword_str = "', '".join([w for w, _ in keywords[:10]])

        rep_str = "\n---\n".join(rep_docs[:3])
        rand_str = "\n---\n".join(rand_docs[:2])

        system_prompt = (
            "You are a clinical informatics expert. "
            "Your task is to create concise, descriptive labels for topics discovered in clinical notes about GLP-1 receptor agonist medications. "
            "Be specific and clinical. Output only the label, nothing else."
        )

        user_prompt = f"""Topic keywords: '{keyword_str}'

Representative clinical notes from this topic:
{rep_str}

Additional notes from this topic:
{rand_str}

Based on the keywords and clinical notes above, create ONE short label (maximum 8 words) that captures the main theme of this topic. Output only the label."""

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
