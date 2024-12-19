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
    """
    df = pd.read_csv(all_predictions_csv)

    # Parse fragment size and residue range from Name
    def parse_name(name):
        """
        Parse the fragment size and residue range from the Name field.
        Handles cases where protein names contain underscores.
        """
        parts = name.split('_')
        try:
            size = int(parts[-2].replace('aa', ''))  # Fragment size
            range_start, range_end = map(int, parts[-1].split('-'))  # Residue range
            return size, range_start, range_end
        except ValueError:
            raise ValueError(f"Unexpected Name format: {name}")

    parsed_data = df['Name'].apply(parse_name)
    df['Fragment_Size'] = parsed_data.apply(lambda x: x[0])
    df['Start'] = parsed_data.apply(lambda x: x[1])
    df['End'] = parsed_data.apply(lambda x: x[2])

    # Expand residue ranges and assign scores
    residue_scores = []
    for _, row in df.iterrows():
        for residue in range(row['Start'], row['End'] + 1):
            residue_scores.append({'Residue': residue, 'LLPS Score': row['LLPS Score'], 'Fragment_Size': row['Fragment_Size']})

    residue_df = pd.DataFrame(residue_scores)

    # Compute average scores for each residue and fragment size
    results = {}
    for size in fragment_sizes:
        size_df = residue_df[residue_df['Fragment_Size'] == size]
        avg_scores = size_df.groupby('Residue')['LLPS Score'].mean().reset_index()
        avg_scores.rename(columns={'LLPS Score': f'{size}aa_Avg_Score'}, inplace=True)
        results[size] = avg_scores

    # Merge scores for all fragment sizes
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
        "--embeddingsFiles", f'{args.fragEmbeddingDir}/*.pt',
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

    # average over the 3 lengths
    merged_scores['Avg_Score'] = merged_scores[[f'{size}aa_Avg_Score' for size in args.probeLengths]].mean(axis=1)

    # plot the avg score
    import matplotlib.pyplot as plt
    residues = merged_scores['Residue']
    plt.figure(figsize=(20, 6))
    plt.plot(residues, merged_scores['Avg_Score'], label='Avg Score', color='tab:blue', lw=3, linestyle='-')
    plt.xlabel('Residue', fontsize=18)
    plt.ylabel('Average LLPS Score', fontsize=18)
    plt.legend(fontsize=22)
    plt.xlim(0, len(sequence))
    plt.ylim(0, 1)
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.title(f'{protein_name}', fontsize=28)
    plt.show()

if __name__ == "__main__":
    main()
