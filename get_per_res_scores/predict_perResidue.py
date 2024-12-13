import argparse
import os
import sys
import subprocess
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def parse_fasta(file_path):
    """Parse a FASTA file and return header and sequence."""
    with open(file_path, 'r') as fasta_file:
        lines = fasta_file.readlines()
    header = lines[0].strip()  # Extract the header
    sequence = ''.join(line.strip() for line in lines[1:])  # Combine sequence lines
    return header, sequence

def generate_all_peptides(header, sequence, fragment_sizes, sliding_window, combined_fasta):
    """
    Generate peptide fragments for all fragment_sizes and write them into a single FASTA file.
    Format: >header_{fragment_size}aa_{start}-{end}
    """
    import numpy as np
    with open(combined_fasta, 'w') as fasta_out:
        for size in fragment_sizes:
            for start in np.arange(0, len(sequence) - size + 1, sliding_window):
                peptide = sequence[start:start + size]
                fasta_out.write(f">{header}_{size}aa_{start+1}-{start+size}\n")
                fasta_out.write(f"{peptide}\n")

def compute_per_residue_scores(all_predictions_csv, fragment_sizes):
    """
    Compute per-residue average scores for each fragment size from a single combined predictions CSV.
    We will:
    - Parse the fragment size from each row's Name field.
    - Group data by fragment size and compute per-residue averages.
    - Merge results for all fragment sizes into a single DataFrame.
    """
    df = pd.read_csv(all_predictions_csv)

    # Parse fragment size from Name. Name format: {header}_{size}aa_{start}-{end}
    # Example: "tau_15aa_1-15"
    df['Fragment_Size'] = df['Name'].apply(lambda x: int(x.split('_')[1].replace('aa', '')))

    # Function to aggregate scores per residue given a subset of df
    def compute_scores_for_size(sub_df, size):
        residue_scores = defaultdict(lambda: {'sum': 0, 'count': 0})
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"Processing {size}aa"):
            name = row['Name']
            score = row['LLPS Score']
            # Extract fragment start and end
            fragment_range = name.split('_')[-1]  # e.g. "1-15"
            start, end = map(int, fragment_range.split('-'))
            for residue in range(start, end + 1):
                residue_scores[residue]['sum'] += score
                residue_scores[residue]['count'] += 1
        averaged_scores = [
            {'Residue': residue, f'{size}aa_Avg_Score': data['sum'] / data['count']}
            for residue, data in residue_scores.items()
        ]
        return pd.DataFrame(averaged_scores)

    # Compute per-res scores for each fragment size
    results = {}
    for size in fragment_sizes:
        sub_df = df[df['Fragment_Size'] == size]
        per_res = compute_scores_for_size(sub_df, size)
        results[size] = per_res

    # Merge all per-res results into a single DataFrame
    merged_scores = None
    for size in fragment_sizes:
        if merged_scores is None:
            merged_scores = results[size]
        else:
            merged_scores = pd.merge(merged_scores, results[size], on='Residue', how='outer')

    merged_scores.sort_values(by='Residue', inplace=True)
    return merged_scores

def main():
    parser = argparse.ArgumentParser(description="Generate per-residue LLPS scores from protein sequence using combined runs.")
    parser.add_argument("fasta", help="Input protein FASTA file.")
    parser.add_argument("-o", "--output", default=None, help="Output CSV file with per-res scores.")
    parser.add_argument("--probeLengths", nargs='*', type=int, default=[15, 25, 40],
                        help="Fragment lengths to probe. Default: 15, 25, 40")
    parser.add_argument("--slidingWindow", type=int, default=1,
                        help="Sliding window size. Default: 1")
    parser.add_argument("--fragEmbeddingDir", default=None,
                        help="Directory where embeddings will be stored. Default: {protein}_fragEmbeddings")
    parser.add_argument("--predictExecutable", default='.',
                        help="Path to predict.py executable. Default: current directory")
    parser.add_argument("--extractExecutable", default='.',
                        help="Path to extract.py executable. Default: current directory")
    parser.add_argument("--LR_model", default="../model_development/LLPS_model_latest.joblib",
                        help="Path to model weights for predict.py. Default: ../model_development/LLPS_model_latest.joblib")

    args = parser.parse_args()

    # Derive protein name from fasta filename
    protein_basename = os.path.basename(args.fasta)
    protein_name = os.path.splitext(protein_basename)[0]

    if args.output is None:
        args.output = f"{protein_name}_perRes_scores.csv"

    if args.fragEmbeddingDir is None:
        args.fragEmbeddingDir = f"{protein_name}_fragEmbeddings"

    # Parse FASTA
    header, sequence = parse_fasta(args.fasta)
    header = header.lstrip(">")

    # Create a single combined FASTA with all fragments
    combined_fasta = f"{protein_name}_fragments.fasta"
    generate_all_peptides(header, sequence, args.probeLengths, args.slidingWindow, combined_fasta)
    print(f"Fragment sequences written to {combined_fasta}")

    # Extract embeddings for all fragments at once
    if not os.path.exists(args.fragEmbeddingDir):
        os.makedirs(args.fragEmbeddingDir, exist_ok=True)

    extract_cmd = [
        sys.executable if args.extractExecutable == '.' else args.extractExecutable,
        "extract.py" if args.extractExecutable == '.' else os.path.join(args.extractExecutable, "extract.py"),
        "esm2_t36_3B_UR50D",
        combined_fasta,
        args.fragEmbeddingDir,
        "--include", "mean"
    ]
    print("Running extract:", " ".join(extract_cmd))
    subprocess.check_call(extract_cmd)

    # Predict once for all fragments
    combined_predictions = f"{protein_name}_fragment_scores.csv"
    predict_cmd = [
        sys.executable if args.predictExecutable == '.' else args.predictExecutable,
        "predict.py" if args.predictExecutable == '.' else os.path.join(args.predictExecutable, "predict.py"),
        "--embeddingsDir", args.fragEmbeddingDir,
        "-o", combined_predictions,
        "--LR_model", args.LR_model
    ]
    print("Running predict:", " ".join(predict_cmd))
    subprocess.check_call(predict_cmd)

    # Compute per-residue scores from the single predictions CSV
    merged_scores = compute_per_residue_scores(combined_predictions, args.probeLengths)

    # Save the merged scores to CSV
    merged_scores.to_csv(args.output, index=False)
    print(f"Per-residue scores written to {args.output}")

if __name__ == "__main__":
    main()
