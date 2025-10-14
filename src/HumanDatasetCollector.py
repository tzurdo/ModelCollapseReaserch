import datasets
import numpy as np
from collections import Counter
import re
import hashlib
import json
from typing import List, Dict

class HumanDatasetCollector:
    def __init__(self):
        self.dataset_stats = {}

    def collect_wikitext(self, max_samples: int = 20) -> List[str]:

        print(f"Loading WikiText-103 dataset (target: {max_samples} samples)...")

        dataset = datasets.load_dataset('wikitext', 'wikitext-103-v1')
        train_data = dataset['train']

        texts = []
        for i, example in enumerate(train_data):
            if len(texts) >= max_samples:
                break

            text = example['text'].strip()

            if self._is_high_quality_text(text):
                cleaned_text = self._clean_text(text)
                if len(cleaned_text) > 100:
                    texts.append(cleaned_text)

        print(f"Collected {len(texts)} high-quality samples\n")

        # Show first 3 samples
        print("=== SAMPLE PREVIEW (First 3) ===")
        for i, sample in enumerate(texts[:3]):
            print(f"\nSample {i+1}:")
            print(sample[:300] + "..." if len(sample) > 300 else sample)
            print("-" * 60)

        return texts

    def _is_high_quality_text(self, text: str) -> bool:

        if len(text) < 50 or len(text) > 5000:
            return False

        words = text.split()
        if len(words) < 10:
            return False

        sentences = text.split('.')
        if not sentences:
            return False

        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return False

        avg_sentence_length = np.mean(sentence_lengths)
        if avg_sentence_length < 3 or avg_sentence_length > 50:
            return False

        word_counts = Counter(words)
        if word_counts:
            most_common_ratio = word_counts.most_common(1)[0][1] / len(words)
            if most_common_ratio > 0.3:
                return False

        return True

    def _clean_text(self, text: str) -> str:

        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'= .* =', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'{{.*?}}', '', text)
        text = re.sub(r'<.*?>', '', text)
        return text.strip()

    def remove_duplicates(self, texts: List[str]) -> List[str]:

        print("Removing duplicates...")
        seen_hashes = set()
        unique_texts = []

        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        print(f"Removed {len(texts) - len(unique_texts)} duplicates")
        return unique_texts

    def assess_quality(self, texts: List[str]) -> Dict:

        if not texts:
            return {'error': 'No texts to assess'}

        all_words = ' '.join(texts).split()

        metrics = {
            'total_samples': len(texts),
            'avg_length_chars': np.mean([len(text) for text in texts]),
            'avg_words': np.mean([len(text.split()) for text in texts]),
            'vocabulary_size': len(set(all_words)),
            'lexical_diversity': len(set(all_words)) / len(all_words) if all_words else 0.0
        }

        return metrics

    def save_dataset(self, texts: List[str], filename: str = "human_baseline.jsonl"):

        with open(filename, 'w') as f:
            for text in texts:
                f.write(json.dumps({"text": text, "source": "wikitext"}) + "\n")
        print(f"\nDataset saved to {filename}")


def main():

    collector = HumanDatasetCollector()

    # Collect WikiText samples
    wikitext_data = collector.collect_wikitext(max_samples=20)

    # Remove duplicates
    unique_data = collector.remove_duplicates(wikitext_data)

    # Show quality metrics
    metrics = collector.assess_quality(unique_data)

    print("\n" + "="*60)
    print("DATASET QUALITY METRICS")
    print("="*60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

    # Save dataset
    collector.save_dataset(unique_data, "Output/human_baseline_dataset.jsonl")

    print(f"\nTotal unique samples: {len(unique_data)}")

    return unique_data


if __name__ == "__main__":
    human_dataset = main()