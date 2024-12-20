import argparse
import re

def calculate_mean_metrics(file_path):
    metrics = {
        'f1': [],
        'loss': [],
        'precision': [],
        'recall': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract metric values using regex
            for key in metrics.keys():
                match = re.search(rf"{key} = ([\d.]+)", line)
                if match:
                    metrics[key].append(float(match.group(1)))
    
    # Calculate mean for each metric
    mean_metrics = {key: sum(values) / len(values) if values else None for key, values in metrics.items()}
    return mean_metrics

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate mean metrics from a text file.")
    parser.add_argument("--file_path", required=True, help="Path to the metrics text file.")
    args = parser.parse_args()

    # Calculate and print the mean metrics
    mean_metrics = calculate_mean_metrics(args.file_path)
    print("Mean Metrics:")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")