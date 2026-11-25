import json
import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from umap import UMAP

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')

DEFAULT_CONFIG = {
    'embeddings': {
        'name': 'BAAI/bge-base-en-v1.5',
        'device': 'cuda'
    },
    'random_state': 42,
    'topic_hdbscan_params': {
        'min_cluster_size': 15,
        'min_samples': 1,
        'metric': 'euclidean',
        'cluster_selection_method': 'eom',
        'prediction_data': True
    },
    'topic_umap_params': {
        'n_neighbors': 15,
        'n_components': 10,
        'min_dist': 0.0,
        'metric': 'cosine',
        'random_state': 42
    },
    'group_umap_params': {
        'n_neighbors': 2,
        'n_components': 3,
        'min_dist': 0.0,
        'metric': 'hellinger',
        'spread': 2,
        'random_state': 42
    },
    'ngram_range': (1, 2),
    'min_text_length': 50,
}


class ClinicalTopicModeling:

    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.df = None
        self.texts = None
        self.embeddings = None
        self.topics = None
        self.topic_model = None
        self.topic_labels = None
        self.groups = None
        self.group_labels = None
        self.c_tf_idf_embed = None
        self.c_tf_idf_vis = None

    def load_data(self, data_path):
        records = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        self.df = pd.DataFrame(records)

        if 'clinical_note' not in self.df.columns:
            raise ValueError("JSONL file must contain a 'clinical_note' column.")

        self.df = self._preprocess_dataframe(self.df)
        self.texts = self.df['clinical_note'].tolist()
        print(f"Loaded {len(self.texts)} clinical notes after preprocessing.")

    @staticmethod
    def _preprocess_dataframe(df):
        df['clinical_note'] = df['clinical_note'].fillna("")
        df['clinical_note'] = df['clinical_note'].str.replace(r'http\S+', '', regex=True)
        df['clinical_note'] = df['clinical_note'].str.replace(r'\n', ' ', regex=True)
        df['clinical_note'] = df['clinical_note'].str.replace(r'\r', ' ', regex=True)
        df['clinical_note'] = df['clinical_note'].str.replace(r'&\w+;', ' ', regex=True)
        df['clinical_note'] = df['clinical_note'].str.replace(r'\s+', ' ', regex=True)
        df['clinical_note'] = df['clinical_note'].str.strip()

        df['note_len'] = df['clinical_note'].str.len()
        min_len = DEFAULT_CONFIG['min_text_length']
        df = df[df['note_len'] >= min_len].reset_index(drop=True)
        df = df.drop(columns=['note_len'])
        return df

    def find_topics(self, save_dir):
        vectorizer_model = CountVectorizer(
            stop_words='english',
            ngram_range=self.config['ngram_range']
        )

        print("Loading embedding model...")
        embedding_model = SentenceTransformer(self.config['embeddings']['name'], device=self.config['embeddings'].get('device', 'cuda'))

        print("Encoding clinical notes...")
        self.embeddings = embedding_model.encode(
            self.texts, show_progress_bar=True, batch_size=64
        )

        print("Running UMAP dimensionality reduction...")
        umap_model = UMAP(**self.config['topic_umap_params'])
        u = umap_model.fit_transform(self.embeddings)

        print("Running HDBSCAN for initial cluster discovery...")
        hdbscan_model = HDBSCAN(**self.config['topic_hdbscan_params'])
        clusters = np.array(hdbscan_model.fit_predict(u))

        n_clusters = np.max(clusters) + 1
        print(f"HDBSCAN found {n_clusters} initial clusters.")

        centroids = np.empty((n_clusters, u.shape[1]))
        for cluster_i in range(n_clusters):
            inds = np.where(clusters == cluster_i)[0]
            centroids[cluster_i, :] = np.mean(u[inds], axis=0)

        print("Refining with KMeans...")
        kmeans_model = KMeans(
            n_clusters=n_clusters,
            init=centroids,
            random_state=42,
            n_init=1
        )

        representation_model = KeyBERTInspired()

        self.topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=kmeans_model,
            representation_model=representation_model,
            verbose=True,
        )

        self.topics, _ = self.topic_model.fit_transform(self.texts, self.embeddings)

        os.makedirs(save_dir, exist_ok=True)
        self.topic_model.save(f'{save_dir}/topic_model.pickle', save_ctfidf=True)
        with open(f'{save_dir}/topics.pickle', 'wb') as fh:
            pickle.dump(self.topics, fh)

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        print(f"Found {n_topics} topics (excluding outlier -1).")
        return self.topics

    def find_representative_docs(self, n_reps=5):
        topics_arr = np.array(self.topics)
        valid_mask = topics_arr != -1
        valid_topics = topics_arr[valid_mask]
        valid_embeddings = self.embeddings[valid_mask]

        unique_topics = np.unique(valid_topics)
        n_topics = len(unique_topics)

        topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
        remapped = np.array([topic_to_idx[t] for t in valid_topics])

        samples_in_topics = [np.where(remapped == i)[0] for i in range(n_topics)]
        centroids = np.array([
            np.mean(valid_embeddings[idx], axis=0) for idx in samples_in_topics
        ])

        rep_docs = []
        for topic_i, (centroid_i, samples_i) in enumerate(zip(centroids, samples_in_topics)):
            embedded = valid_embeddings[samples_i]
            dists = cosine_distances(embedded, centroid_i.reshape(1, -1)).flatten()
            sorted_inds = np.argsort(dists)[:n_reps]
            original_inds = valid_mask.nonzero()[0][samples_i[sorted_inds]]
            rep_docs.append(self.df.iloc[original_inds]['clinical_note'].tolist())

        return rep_docs

    def find_random_docs(self, n_reps=5):
        topics_arr = np.array(self.topics)
        valid_mask = topics_arr != -1
        valid_topics = topics_arr[valid_mask]
        unique_topics = np.unique(valid_topics)

        rand_docs = []
        for topic in unique_topics:
            topic_inds = np.where(valid_topics == topic)[0]
            original_inds = valid_mask.nonzero()[0][topic_inds]
            if len(original_inds) < n_reps:
                samples = self.df.iloc[original_inds]['clinical_note'].tolist()
            else:
                sampled = np.random.choice(original_inds, size=n_reps, replace=False)
                samples = self.df.iloc[sampled]['clinical_note'].tolist()
            rand_docs.append(samples)

        return rand_docs

    def find_groups(self, save_dir, llim=3, ulim=30):
        topics_dict = self.topic_model.get_topics()
        valid_topics = sorted([t for t in topics_dict.keys() if t != -1])
        n_topics = len(valid_topics)

        all_words = set()
        for t in valid_topics:
            for word, _ in topics_dict[t]:
                all_words.add(word)
        word_to_idx = {w: i for i, w in enumerate(sorted(all_words))}

        topic_vectors = np.zeros((n_topics, len(word_to_idx)))
        for i, t in enumerate(valid_topics):
            for word, score in topics_dict[t]:
                if word in word_to_idx:
                    topic_vectors[i, word_to_idx[word]] = score

        c_tf_idf_mms = MinMaxScaler().fit_transform(topic_vectors)

        self.c_tf_idf_vis = UMAP(
            n_neighbors=2, n_components=2, metric='hellinger',
            random_state=self.config['random_state']
        ).fit_transform(c_tf_idf_mms)
        self.c_tf_idf_vis = np.nan_to_num(self.c_tf_idf_vis, nan=0.0)

        self.c_tf_idf_embed = UMAP(
            **self.config['group_umap_params']
        ).fit_transform(c_tf_idf_mms)
        self.c_tf_idf_embed = np.nan_to_num(self.c_tf_idf_embed, nan=0.0)

        n_topics = self.c_tf_idf_embed.shape[0]
        if ulim > n_topics:
            ulim = n_topics

        print("Finding optimal number of groups via Silhouette score...")
        ss_scores = []
        cluster_range = np.arange(llim, ulim + 1, 1)

        for n_clusters in tqdm(cluster_range):
            n_neighbors = min(10, n_topics - 1) if n_topics > 1 else 2
            clusters = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                n_components=2,
                affinity='nearest_neighbors',
                n_neighbors=n_neighbors
            ).fit_predict(self.c_tf_idf_embed)
            ss_scores.append(silhouette_score(self.c_tf_idf_embed, clusters))

        best_idx = np.argmax(ss_scores)
        ideal_n = cluster_range[best_idx]
        print(f"Best silhouette score: {ss_scores[best_idx]:.3f} at n_clusters={ideal_n}")

        n_neighbors = min(10, n_topics - 1) if n_topics > 1 else 2
        self.groups = SpectralClustering(
            n_clusters=ideal_n,
            random_state=self.config['random_state'],
            n_components=2,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors
        ).fit_predict(self.c_tf_idf_embed) + 1

        with open(f'{save_dir}/silhouette_scores.pkl', 'wb') as fh:
            pickle.dump({'cluster_range': cluster_range, 'scores': ss_scores}, fh)

        return self.groups

    def plot_groups(self, save_dir, group_labels_dict=None):
        plt.figure(figsize=(10, 5))

        vis_arr = self.c_tf_idf_vis
        topic_info = self.topic_model.get_topic_info()

        labels_for_plot = []
        if group_labels_dict:
            for g in self.groups:
                labels_for_plot.append(group_labels_dict.get(g, f"Group {g}"))
        else:
            labels_for_plot = [f"Group {g}" for g in self.groups]

        ax = sns.scatterplot(
            x=vis_arr[:, 0], y=vis_arr[:, 1],
            size=topic_info['Count'],
            hue=labels_for_plot,
            sizes=(100, 5000),
            alpha=0.5, palette='tab20', legend=True, edgecolor='k'
        )

        ax.set_title('Topics, Grouped by Similarity of Content', fontsize=16, pad=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.figure.savefig(f'{save_dir}/figure_groups.png', dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved group plot to {save_dir}/figure_groups.png")

    def plot_silhouette(self, save_dir):
        with open(f'{save_dir}/silhouette_scores.pkl', 'rb') as fh:
            data = pickle.load(fh)

        plt.figure(figsize=(6, 3))
        sns.lineplot(x=data['cluster_range'], y=data['scores'], marker='o')
        plt.ylabel('Silhouette Score')
        plt.xlabel('Number of Groups')
        plt.title('Clustering Performance')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/silhouette_curve.png', dpi=300, bbox_inches="tight")
        plt.close()

    def create_topic_table(self, save_dir, topic_labels_dict=None, group_labels_dict=None):
        rep_docs = self.find_representative_docs(n_reps=1)

        valid_topics = sorted([t for t in set(self.topics) if t != -1])
        topic_to_idx = {t: i for i, t in enumerate(valid_topics)}

        rows = []
        for t in valid_topics:
            idx = topic_to_idx[t]
            group = int(self.groups[idx]) if self.groups is not None else None
            label = topic_labels_dict.get(t, '') if topic_labels_dict else ''
            group_label = group_labels_dict.get(group, '') if (group_labels_dict and group) else ''
            count = int(np.sum(np.array(self.topics) == t))
            rep_doc = rep_docs[idx][0] if rep_docs and idx < len(rep_docs) else ''

            rows.append({
                'Topic_ID': t,
                'Group_ID': group,
                'Group_Label': group_label,
                'Topic_Label': label,
                'Note_Count': count,
                'Representative_Note': rep_doc[:500]
            })

        topic_table = pd.DataFrame(rows)
        topic_table.to_excel(f'{save_dir}/topic_table.xlsx', index=False)
        print(f"Saved topic table to {save_dir}/topic_table.xlsx")
        return topic_table
