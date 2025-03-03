#!/usr/bin/env python3

import re
import argparse

def extract_languages(file_path):
    # List to store all found languages
    languages = []
    
    # Regex pattern to match language=xyz format
    pattern = r'language=([^\s]+)'
    
    try:
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Find all language matches
            matches = re.findall(pattern, content)
            
            # Add matches to our languages list
            languages.extend(matches)
            
            # Print all found languages as a comma-separated string
            languages_string = ", ".join(languages)
            print(f"Languages found: {languages_string}")
            
            # Print the count of languages
            print(f"Total number of languages: {len(languages)}")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract languages from a text file.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the text file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract languages from the specified file
    extract_languages(args.file_path)