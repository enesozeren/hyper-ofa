import os
import argparse
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for the progress bar

DATASET_NAMES = ["eng_Latn", # English for preventing catastrophic forgetting
                 "tur_Latn", "ell_Grek", "bul_Cyrl", "ces_Latn", "kor_Hang", # High resource languages
                 "jpn_Japn", "yue_Hani", "uig_Arab", "pam_Latn", "gla_Latn", # Mid resource languges
                 "mhr_Cyrl", "swh_Latn", "cmn_Hani", "pes_Arab", "dtp_Latn"] # Low resource languges

def create_dataset(dataset_names: list, output_path: str, batch_size: int, cache_dir:str):
    """
    Create a merged dataset file for HyperOFA continued pre-training.

    Args:
        dataset_names (list): List of dataset names to merge.
        output_path (str): Path to save the output file.
        batch_size (int): Number of sentences to write to the file in each batch.
        cache_dir (str): Cache dir for load_dataset
    """
    # File to save the merged sentences
    output_file = os.path.join(output_path, "hyperofa_training_data.txt")

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for dataset_name in dataset_names:
            dataset = load_dataset("cis-lmu/Glot500", dataset_name, split="train", cache_dir=cache_dir)
            # Print the dataset length
            print(f"{dataset_name}: {len(dataset)}")

            # Batch processing
            batch = []
            for example in tqdm(dataset, desc=f"Writing {dataset_name}", unit="examples"):
                sentence = example["text"].strip()  # Replace "text" with the appropriate field
                batch.append(sentence)

                # Write batch to file
                if len(batch) == batch_size:
                    f.write("\n".join(batch) + "\n")
                    batch = []  # Clear the batch

            # Write any remaining sentences in the last batch
            if batch:
                f.write("\n".join(batch) + "\n")

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
    args = parser.parse_args()    

    create_dataset(DATASET_NAMES, args.output_path, args.batch_size, args.cache_dir)


if __name__ == "__main__":
    main()