import argparse

def parse_accuracy_file(file_path):
    """
    Parse the accuracy file and return a dictionary of language accuracies.
    """
    results = {}
    current_language = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or separator lines
            if not line or line.startswith('==='):
                continue
                
            # Parse language line
            if line.startswith('language='):
                current_language = line.split('=')[1]
                results[current_language] = {}
                continue
                
            # Parse accuracy lines
            if line.startswith('Acc'):
                metric, value = line.split('=')
                metric = metric.strip()
                value = float(value.strip())
                if current_language:
                    results[current_language][metric] = value
                    
    return results

def calculate_averages(results, target_languages):
    """
    Calculate average accuracies for specified languages.
    """
    found_languages = []
    acc1_sum = acc5_sum = acc10_sum = 0
    
    for lang in target_languages:
        if lang in results:
            found_languages.append(lang)
            acc1_sum += results[lang].get('Acc1', 0)
            acc5_sum += results[lang].get('Acc5', 0)
            acc10_sum += results[lang].get('Acc10', 0)
    
    if not found_languages:
        return None
        
    num_langs = len(found_languages)
    return {
        'Acc1': acc1_sum / num_langs,
        'Acc5': acc5_sum / num_langs,
        'Acc10': acc10_sum / num_langs,
        'languages_found': found_languages,
        'total_languages': num_langs
    }

def main():
    # Define target languages
    DATASET_NAMES = [
        "eng_Latn",
        "tur_Latn", "ell_Grek", "bul_Cyrl", "ces_Latn", "kor_Hang",
        "zsm_Latn", "kat_Geor", "fry_Latn", "khm_Khmr", "jpn_Japn",
        "yue_Hani", "tuk_Latn", "uig_Arab", "pam_Latn", "kab_Latn", "gla_Latn",
        "mhr_Cyrl", "swh_Latn", "cmn_Hani", "pes_Arab", "dtp_Latn"
    ]
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate average accuracies for specific languages.')
    parser.add_argument('--file_path', help='Path to the accuracy results file')
    args = parser.parse_args()
    
    # Parse file and calculate averages
    try:
        results = parse_accuracy_file(args.file_path)
        averages = calculate_averages(results, DATASET_NAMES)
        
        if averages:
            print(f"\nResults for {averages['total_languages']} languages:")
            print(f"Average Acc@1: {averages['Acc1']:.4f}")
            print(f"Average Acc@5: {averages['Acc5']:.4f}")
            print(f"Average Acc@10: {averages['Acc10']:.4f}")
            print("\nLanguages found:", ', '.join(averages['languages_found']))
            
            missing_langs = set(DATASET_NAMES) - set(averages['languages_found'])
            if missing_langs:
                print("\nMissing languages:", ', '.join(missing_langs))
        else:
            print("No target languages found in the input file.")
            
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()