import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import List, Dict
import random


try:
    HF_TOKEN = "my_token"
    login(token=HF_TOKEN)
    print("Authenticated with Hugging Face\n")
except:
    print("No HF token found. Using public models only.\n")

class AIDatasetGenerator:

    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Loading model: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model for CPU (no quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,            # Don't use device mapping
            low_cpu_mem_usage=True
        )

        print("Model loaded successfully!")

    def generate_from_prompts(self,
                             prompts: List[str],
                             samples_per_prompt: int = 1,
                             max_length: int = 200,
                             temperature: float = 0.8) -> List[str]:

        generated_texts = []

        print(f"Generating {len(prompts) * samples_per_prompt} AI samples...")

        for i, prompt in enumerate(prompts):
            for j in range(samples_per_prompt):
                try:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Remove the prompt from output
                    generated_text = generated_text[len(prompt):].strip()

                    if len(generated_text) > 25:  # Minimum length check
                        generated_texts.append(generated_text)
                    else:
                        print(f"Dropping text: {generated_text}")

                    if (i * samples_per_prompt + j + 1) % 10 == 0:
                        print(f"Generated {len(generated_texts)} samples...")

                except Exception as e:
                    print(f"Error generating sample: {e}")
                    continue

        print(f"Total AI samples generated: {len(generated_texts)}\n")
        return generated_texts

    def create_prompts_from_human_data(self,
                                       human_dataset_file: str,
                                       num_prompts: int = 100) -> List[str]:

        print(f"Creating prompts from {human_dataset_file}...")

        prompts = []
        with open(human_dataset_file, 'r') as f:
            human_samples = [json.loads(line) for line in f]

        # Sample random texts
        sampled = random.sample(human_samples, min(num_prompts, len(human_samples)))

        for sample in sampled:
          text = sample['text']
          words = text.split()

          # Skip if text is too short
          if len(words) < 20:
              continue

          # Take first 20-50 words as prompt
          prompt_length = random.randint(20, min(50, len(words)))
          prompt = ' '.join(words[:prompt_length])
          prompts.append(prompt)

        print(f"Created {len(prompts)} prompts\n")
        return prompts

    def assess_ai_quality(self, texts: List[str]) -> Dict:
        """Calculate quality metrics for AI-generated text"""
        import numpy as np
        from collections import Counter

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

    def save_ai_dataset(self, texts: List[str], filename: str):

        # clean_model_name = self.model_name.split('/')[-1].replace('-', '_').lower()
        # filename = f"ai_generated_{clean_model_name}_dataset.jsonl"

        with open(filename, 'w') as f:
            for text in texts:
                f.write(json.dumps({
                    "text": text,
                    "source": "ai_generated",
                    "model": self.model_name
                }) + "\n")
        print(f"AI dataset saved to {filename}")


def main():
    """Main execution for AI dataset generation"""

    # MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    # MODEL_NAME = "EleutherAI/gpt-j-6b"
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    # MODEL_NAME = "gpt2-medium"

    print(f"Using model: {MODEL_NAME}\n")

    generator = AIDatasetGenerator(model_name=MODEL_NAME)

    # Generate from human dataset prompts
    print("Generating from human dataset prompts...")
    print("-"*60)
    prompts = generator.create_prompts_from_human_data(
        human_dataset_file="Output/human_baseline_dataset.jsonl",
        num_prompts=20
    )

    # Generate AI samples
    ai_texts = generator.generate_from_prompts(
        prompts=prompts,
        samples_per_prompt=1,
        max_length=150,
        temperature=0.8
    )

    # Show samples
    print("\n" + "="*60)
    print("AI-GENERATED SAMPLES (First 3)")
    print("="*60)
    for i, sample in enumerate(ai_texts[:3]):
        print(f"\nAI Sample {i+1}:")
        print(sample[:300] + "..." if len(sample) > 300 else sample)
        print("-"*60)

    # Assess quality
    metrics = generator.assess_ai_quality(ai_texts)

    print("\n" + "="*60)
    print("AI DATASET QUALITY METRICS")
    print("="*60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

    # Save dataset
    saved_filename = generator.save_ai_dataset(ai_texts, f"Output/ai_generated_{MODEL_NAME}_dataset.jsonl")

    print(f"\nTotal AI samples: {len(ai_texts)}")
    print(f"Saved to: {saved_filename}")

    # # Download to local machine (Colab only)
    # try:
    #     from google.colab import files
    #     files.download(saved_filename)
    #     print(f"\n✓ File '{saved_filename}' downloaded to your local machine!")
    # except:
    #     print(f"\nNot running in Colab - file saved in current directory")

    return ai_texts


if __name__ == "__main__":
    ai_dataset = main()