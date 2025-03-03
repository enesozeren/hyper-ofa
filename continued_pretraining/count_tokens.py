import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm

def count_tokens(file_path, tokenizer, chunk_size=1024*1024):
    """
    Count tokens in a large text file by processing it in chunks.
    
    Args:
        file_path (str): Path to the text file
        tokenizer: Hugging Face tokenizer
        chunk_size (int): Size of chunks to read in bytes
    
    Returns:
        int: Total number of tokens in the file
    """
    total_tokens = 0
    file_size = os.path.getsize(file_path)
    
    # Create a progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing") as pbar:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            # Buffer to hold text that might be split between chunks
            buffer = ""
            
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                
                # Add the previous buffer to the current chunk
                text = buffer + chunk
                
                # If we're not at the end of the file, find the last newline
                if len(chunk) == chunk_size:
                    last_newline = text.rfind('\n')
                    if last_newline != -1:
                        # Process complete lines
                        process_text = text[:last_newline]
                        # Save the remainder for the next iteration
                        buffer = text[last_newline:]
                    else:
                        # If no newline found, process the whole chunk
                        process_text = text
                        buffer = ""
                else:
                    # At the end of the file, process everything
                    process_text = text
                    buffer = ""
                
                # Count tokens in the current chunk
                tokens = tokenizer(process_text)["input_ids"]
                total_tokens += len(tokens)
                
                # Update progress bar
                pbar.update(len(chunk.encode('utf-8')))
    
    # Process any remaining text in the buffer
    if buffer:
        tokens = tokenizer(buffer)["input_ids"]
        total_tokens += len(tokens)
    
    return total_tokens

def main():
    parser = argparse.ArgumentParser(description='Count tokens in a large text file')
    parser.add_argument('--file_path', type=str, help='Path to the text file')
    parser.add_argument('--chunk_size', type=int, default=64*64, 
                        help='Size of chunks to read in bytes')
    args = parser.parse_args()
    
    print(f"Loading the cis-lmu/glot500-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    
    print(f"Counting tokens in {args.file_path}...")
    token_count = count_tokens(args.file_path, tokenizer, args.chunk_size)
    
    print(f"Total tokens: {token_count:,}")

if __name__ == "__main__":
    main()