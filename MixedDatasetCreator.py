import json
import random
import numpy as np
from collections import Counter
from typing import List, Dict

class MixedDatasetCreator:
    def __init__(self):
        self.dataset_stats = {}

    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load data from JSONL file"""
        print(f"Loading data from {filename}...")
        
        data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            print(f"Loaded {len(data)} samples from {filename}")
        except FileNotFoundError:
            print(f"Error: File {filename} not found!")
            return []
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
        
        return data

    def create_mixed_dataset(self, 
                           human_file: str, 
                           ai_file: str, 
                           target_total: int = 20) -> List[Dict]:
        """
        Create mixed dataset with 50% human and 50% AI samples
        
        Args:
            human_file: Path to human baseline JSONL file
            ai_file: Path to AI generated JSONL file  
            target_total: Total number of samples in mixed dataset
        """
        print("="*60)
        print("CREATING MIXED DATASET (50% Human + 50% AI)")
        print("="*60)
        
        # Load both datasets
        human_data = self.load_jsonl(human_file)
        ai_data = self.load_jsonl(ai_file)
        
        if not human_data or not ai_data:
            print("Cannot create mixed dataset - missing source files!")
            return []
        
        # Calculate samples needed from each source
        samples_per_source = target_total // 2
        
        print(f"\nTarget: {target_total} total samples")
        print(f"Taking {samples_per_source} samples from each source")
        
        # Sample from each dataset
        human_samples = random.sample(human_data, min(samples_per_source, len(human_data)))
        ai_samples = random.sample(ai_data, min(samples_per_source, len(ai_data)))
        
        # Add source labels for tracking
        for sample in human_samples:
            sample['mixed_source'] = 'human'
            
        for sample in ai_samples:
            sample['mixed_source'] = 'ai'
        
        # Combine and shuffle
        mixed_dataset = human_samples + ai_samples
        random.shuffle(mixed_dataset)
        
        print(f"\nMixed dataset created:")
        print(f"- Human samples: {len(human_samples)}")
        print(f"- AI samples: {len(ai_samples)}")
        print(f"- Total samples: {len(mixed_dataset)}")
        
        return mixed_dataset

    def show_dataset_preview(self, dataset: List[Dict], num_samples: int = 3):
        """Show preview of mixed dataset"""
        print("\n" + "="*60)
        print(f"MIXED DATASET PREVIEW (First {num_samples} samples)")
        print("="*60)
        
        for i, sample in enumerate(dataset[:num_samples]):
            source_type = sample.get('mixed_source', 'unknown')
            original_source = sample.get('source', 'unknown')
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Source Type: {source_type.upper()} (originally from {original_source})")
            print(f"Text: {sample['text'][:200]}...")
            print("-"*60)

    def assess_mixed_quality(self, dataset: List[Dict]) -> Dict:
        """Calculate quality metrics for mixed dataset"""
        if not dataset:
            return {'error': 'No data to assess'}
        
        # Extract just the text for analysis
        texts = [sample['text'] for sample in dataset]
        all_words = ' '.join(texts).split()
        
        # Count sources
        human_count = sum(1 for sample in dataset if sample.get('mixed_source') == 'human')
        ai_count = sum(1 for sample in dataset if sample.get('mixed_source') == 'ai')
        
        metrics = {
            'total_samples': len(dataset),
            'human_samples': human_count,
            'ai_samples': ai_count,
            'human_percentage': (human_count / len(dataset)) * 100,
            'ai_percentage': (ai_count / len(dataset)) * 100,
            'avg_length_chars': np.mean([len(text) for text in texts]),
            'avg_words': np.mean([len(text.split()) for text in texts]),
            'vocabulary_size': len(set(all_words)),
            'lexical_diversity': len(set(all_words)) / len(all_words) if all_words else 0.0
        }
        
        return metrics

    def save_mixed_dataset(self, dataset: List[Dict], filename: str = "mixed_dataset.jsonl"):
        """Save mixed dataset to JSONL file"""
        
        # Update source to indicate mixed nature
        for sample in dataset:
            sample['source'] = 'mixed_50_50'
        
        with open(filename, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + "\n")
        
        print(f"\nMixed dataset saved to {filename}")
        return filename

def main():
    """Main execution for mixed dataset creation"""
    
    print("="*60)
    print("MIXED DATASET CREATION")
    print("For Model Collapse Research")
    print("="*60 + "\n")
    
    creator = MixedDatasetCreator()
    
    # Create mixed dataset
    mixed_dataset = creator.create_mixed_dataset(
        human_file="Output/human_baseline_dataset.jsonl",
        ai_file="Output/ai_generated_gpt2-medium_dataset.jsonl", 
        target_total=20  # You specified 20 samples
    )
    
    if not mixed_dataset:
        print("Failed to create mixed dataset!")
        return None
    
    # Show preview
    creator.show_dataset_preview(mixed_dataset, num_samples=3)
    
    # Assess quality
    metrics = creator.assess_mixed_quality(mixed_dataset)
    
    print("\n" + "="*60)
    print("MIXED DATASET QUALITY METRICS")
    print("="*60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    # Save dataset
    saved_filename = creator.save_mixed_dataset(mixed_dataset, "Output/mixed_gpt2-medium_dataset.jsonl")
    
    print(f"\n✓ Mixed dataset ready for model collapse research!")
    print(f"Now you have all three datasets:")
    print(f"  - Human only: human_baseline_dataset.jsonl")
    print(f"  - AI only: ai_generated_gpt2_medium_dataset.jsonl")  
    print(f"  - Mixed 50/50: {saved_filename}")
    
    return mixed_dataset

if __name__ == "__main__":
    mixed_dataset = main()