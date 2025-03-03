import argparse

def parse_metrics_file(file_path):
    """
    Parse the metrics file and return a dictionary of language metrics.
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
                
            # Parse metric lines (Acc or f1/precision/recall/loss)
            if '=' in line:
                metric, value = line.split('=')
                metric = metric.strip()
                value = float(value.strip())
                if current_language:
                    results[current_language][metric] = value
                    
    return results

def calculate_accuracy_averages(results, target_languages):
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

def calculate_f1_averages(results, target_languages):
    """
    Calculate average F1 scores and other metrics for specified languages.
    """
    found_languages = []
    f1_sum = precision_sum = recall_sum = loss_sum = 0
    metrics_found = {'f1': 0, 'precision': 0, 'recall': 0, 'loss': 0}
    
    for lang in target_languages:
        if lang in results:
            found_languages.append(lang)
            
            if 'f1' in results[lang]:
                f1_sum += results[lang]['f1']
                metrics_found['f1'] += 1
                
            if 'precision' in results[lang]:
                precision_sum += results[lang]['precision']
                metrics_found['precision'] += 1
                
            if 'recall' in results[lang]:
                recall_sum += results[lang]['recall']
                metrics_found['recall'] += 1
                
            if 'loss' in results[lang]:
                loss_sum += results[lang]['loss']
                metrics_found['loss'] += 1
    
    if not found_languages:
        return None
        
    num_langs = len(found_languages)
    averages = {
        'languages_found': found_languages,
        'total_languages': num_langs
    }
    
    # Only add metrics that were found in at least one language
    if metrics_found['f1'] > 0:
        averages['f1'] = f1_sum / metrics_found['f1']
    if metrics_found['precision'] > 0:
        averages['precision'] = precision_sum / metrics_found['precision']
    if metrics_found['recall'] > 0:
        averages['recall'] = recall_sum / metrics_found['recall']
    if metrics_found['loss'] > 0:
        averages['loss'] = loss_sum / metrics_found['loss']
        
    return averages

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
    parser = argparse.ArgumentParser(description='Calculate average metrics for specific languages.')
    parser.add_argument('--file_path', required=True, help='Path to the metrics results file')
    parser.add_argument('--metric', choices=['acc', 'f1'], default='acc', 
                        help='Metric type to calculate: accuracy (acc) or F1 score (f1)')
    args = parser.parse_args()
    
    # Parse file and calculate averages based on metric type
    try:
        results = parse_metrics_file(args.file_path)
        
        if args.metric == 'acc':
            averages = calculate_accuracy_averages(results, DATASET_NAMES)
            
            if averages:
                print(f"\nResults for {averages['total_languages']} languages:")
                print(f"Average Acc@1: {averages['Acc1']:.4f}")
                print(f"Average Acc@5: {averages['Acc5']:.4f}")
                print(f"Average Acc@10: {averages['Acc10']:.4f}")
            else:
                print("No accuracy metrics found in the input file.")
                
        elif args.metric == 'f1':
            averages = calculate_f1_averages(results, DATASET_NAMES)
            
            if averages:
                print(f"\nResults for {averages['total_languages']} languages:")
                if 'f1' in averages:
                    print(f"Average F1 Score: {averages['f1']:.4f}")
                if 'precision' in averages:
                    print(f"Average Precision: {averages['precision']:.4f}")
                if 'recall' in averages:
                    print(f"Average Recall: {averages['recall']:.4f}")
                if 'loss' in averages:
                    print(f"Average Loss: {averages['loss']:.4f}")
            else:
                print("No F1 metrics found in the input file.")
        
        # Print language information
        if averages:
            print("\nLanguages found:", ', '.join(averages['languages_found']))
            
            missing_langs = set(DATASET_NAMES) - set(averages['languages_found'])
            if missing_langs:
                print("\nMissing languages:", ', '.join(missing_langs))
            
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()