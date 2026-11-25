import argparse
import os
import json
import sys

from topic_model import ClinicalTopicModeling
from label_topics import TopicLabeler
from find_groups import GroupLabeler


def get_data_paths(base_dir):
    train_path = os.path.join(base_dir, "finetune", "training_data", "train_test", "train.jsonl")
    test_path = os.path.join(base_dir, "finetune", "training_data", "train_test", "test.jsonl")
    return train_path, test_path


def combine_jsonl(train_path, test_path, output_path):
    records = []
    for path in [train_path, test_path]:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)

    with open(output_path, 'w') as f:
        for record in records:
            f.write(record + "\n")

    return output_path


def run_pipeline(data_path, output_dir, vllm_url, model_name, min_cluster_size=15):
    os.makedirs(output_dir, exist_ok=True)

    config = {
        'embeddings': {
            'name': 'BAAI/bge-base-en-v1.5',
            'device': 'cuda'
        },
        'random_state': 42,
        'topic_hdbscan_params': {
            'min_cluster_size': min_cluster_size,
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

    tm = ClinicalTopicModeling(config=config)

    print(f"\n{'='*60}")
    print(f"Loading data from: {data_path}")
    print(f"{'='*60}\n")
    tm.load_data(data_path)

    print(f"\n{'='*60}")
    print("STEP 1: Finding topics (embedding + clustering)")
    print(f"{'='*60}\n")
    tm.find_topics(output_dir)

    print(f"\n{'='*60}")
    print("STEP 2: Labeling topics via vLLM")
    print(f"{'='*60}\n")
    labeler = TopicLabeler(tm, vllm_url=vllm_url, model_name=model_name)
    topic_labels = labeler.label_topics(output_dir)

    print(f"\n{'='*60}")
    print("STEP 3: Finding groups (Spectral Clustering)")
    print(f"{'='*60}\n")
    tm.find_groups(output_dir)
    tm.plot_silhouette(output_dir)

    print(f"\n{'='*60}")
    print("STEP 4: Labeling groups via vLLM")
    print(f"{'='*60}\n")
    group_labeler = GroupLabeler(tm, vllm_url=vllm_url, model_name=model_name)
    group_labels = group_labeler.label_groups(output_dir, topic_labels_dict=topic_labels)

    print(f"\n{'='*60}")
    print("STEP 5: Visualization and summary")
    print(f"{'='*60}\n")
    tm.plot_groups(output_dir, group_labels_dict=group_labels)
    tm.create_topic_table(output_dir, topic_labels_dict=topic_labels, group_labels_dict=group_labels)

    print(f"\n{'='*60}")
    print(f"Pipeline complete! Results saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    workspace_root = "/home/user/GLP_project/GLP_project_ad/code"
    base_results_dir = "/home/user/GLP_project/GLP_project_ad/code/topic_modeling/results"

    parser = argparse.ArgumentParser(description="Clinical Notes Topic Modeling Pipeline")
    parser.add_argument(
        '--data',
        type=str,
        choices=['train', 'test', 'combined'],
        default='train',
        help='Which dataset to use: train, test, or combined'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: results/<data>)'
    )
    parser.add_argument(
        '--vllm_url',
        type=str,
        default='http://localhost:8091',
        help='vLLM server URL'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/gemma-4-31B-it',
        help='Model name for vLLM'
    )
    parser.add_argument(
        '--min_cluster_size',
        type=int,
        default=15,
        help='Minimum cluster size for HDBSCAN'
    )

    args = parser.parse_args()

    train_path, test_path = get_data_paths(workspace_root)

    if args.data == 'train':
        data_path = train_path
    elif args.data == 'test':
        data_path = test_path
    elif args.data == 'combined':
        combined_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined.jsonl')
        data_path = combine_jsonl(train_path, test_path, combined_path)
    else:
        raise ValueError(f"Unknown data option: {args.data}")

    if args.output_dir is None:
        args.output_dir = os.path.join(base_results_dir, args.data)

    run_pipeline(
        data_path=data_path,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        min_cluster_size=args.min_cluster_size
    )


if __name__ == '__main__':
    main()
