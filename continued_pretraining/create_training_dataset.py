import os
import shutil
import argparse
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for the progress bar

DATASET_NAMES = ["eng_Latn",  # English for preventing catastrophic forgetting (>5M sent)
                 "tur_Latn", "ell_Grek", "bul_Cyrl", "ces_Latn", "kor_Hang",  # High resource languages (>5M sent)
                 "zsm_Latn", "kat_Geor", "fry_Latn", "khm_Khmr", "jpn_Japn", # Mid resource languges (>500K sent)
                 "yue_Hani", "tuk_Latn", "uig_Arab", "pam_Latn", "kab_Latn", "gla_Latn",  # Low resource languges (<500K sent)
                 "mhr_Cyrl", "swh_Latn", "cmn_Hani", "pes_Arab", "dtp_Latn"]  # Low resource languges (<500K sent)

def clear_dataset_cache(cache_dir: str, dataset_name: str):
    """
    Clear the cache for a specific dataset.
    Args:
        cache_dir (str): Base cache directory
        dataset_name (str): Name of the dataset to clear
    """
    # The dataset cache is typically stored in a subdirectory structure
    dataset_cache_path = os.path.join(cache_dir, "cis-lmu___glot500", dataset_name)
    if os.path.exists(dataset_cache_path):
        print(f"Clearing cache for {dataset_name}...")
        shutil.rmtree(dataset_cache_path)
        print(f"Cache cleared for {dataset_name}")

def create_dataset(dataset_names: list, output_path: str, batch_size: int, cache_dir: str, max_sentences: int):
    """
    Create a merged dataset file for HyperOFA continued pre-training.
    Args:
        dataset_names (list): List of dataset names to merge.
        output_path (str): Path to save the output file.
        batch_size (int): Number of sentences to write to the file in each batch.
        cache_dir (str): Cache dir for load_dataset
        max_sentences (int): Maximum number of sentences to process per dataset
    """
    # File to save the merged sentences
    output_file = os.path.join(output_path, "hyperofa_training_data.txt")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for dataset_name in dataset_names:
            try:
                dataset = load_dataset("cis-lmu/Glot500", dataset_name, split="train", cache_dir=cache_dir)
                # Print the dataset length
                print(f"{dataset_name}: {len(dataset)}")
                
                # Batch processing
                batch = []
                sentence_count = 0
                
                # Create progress bar with total being min of dataset length and max_sentences
                total_sentences = min(len(dataset), max_sentences)
                
                for example in tqdm(dataset, desc=f"Writing {dataset_name}", total=total_sentences, unit="examples"):
                    if sentence_count >= max_sentences:
                        print(f"Reached {max_sentences} sentences limit for {dataset_name}, moving to next dataset")
                        break
                        
                    sentence = example["text"].strip()  # Replace "text" with the appropriate field
                    batch.append(sentence)
                    sentence_count += 1
                    
                    # Write batch to file
                    if len(batch) == batch_size:
                        f.write("\n".join(batch) + "\n")
                        batch = []  # Clear the batch
                        
                # Write any remaining sentences in the last batch
                if batch:
                    f.write("\n".join(batch) + "\n")
                
                print(f"Processed {sentence_count} sentences from {dataset_name}")
                
            finally:
                # Clear the cache for this dataset before moving to the next one
                clear_dataset_cache(cache_dir, dataset_name)

def main():
    parser = argparse.ArgumentParser(
        description='Create dataset file for HyperOFA continued pre-training')
    parser.add_argument('--output_path', type=str,
                       default='continued_pretraining/dataset',
                       help='Output path')
    parser.add_argument('--cache_dir', type=str,
                       default="caches",
                       help='Cache directory for huggingface load_dataset')
    parser.add_argument('--batch_size', type=int,
                       default=10_000,
                       help='Batch size for writing to the file')
    parser.add_argument('--max_sentences', type=int,
                       default=5_000_000,
                       help='Maximum number of sentences to process per dataset')
    args = parser.parse_args()
    
    create_dataset(DATASET_NAMES, args.output_path, args.batch_size, args.cache_dir, args.max_sentences)

if __name__ == "__main__":
    main()