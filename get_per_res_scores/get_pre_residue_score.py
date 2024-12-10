#%%
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def compute_per_residue_scores(scores_csv, fragment_size):
    """
    Computes per-residue scores from fragment scores for a given fragment size.

    Args:
    - scores_csv (str): The input CSV file containing fragment scores.
    - fragment_size (int): Size of the peptide fragments (for naming purposes).

    Returns:
    - pd.DataFrame: DataFrame with averaged per-residue scores.
    """
    # Load the CSV file with fragment scores
    scores_df = pd.read_csv(scores_csv)

    # Dictionary to aggregate scores for each residue
    residue_scores = defaultdict(lambda: {'sum': 0, 'count': 0})

    # Iterate through each row in the CSV file
    for index, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc=f"Processing {fragment_size}aa"):
        fragment_name = row['Name']  # Example: "tau_10aa_1-10"
        score = row['LLPS Score']  # LLPS Score

        # Extract start and end positions from fragment name
        fragment_range = fragment_name.split('_')[-1]  # "1-10"
        start, end = map(int, fragment_range.split('-'))

        # Update scores for each residue in the fragment range
        for residue in range(start, end + 1):
            residue_scores[residue]['sum'] += score
            residue_scores[residue]['count'] += 1

    # Compute the average score for each residue
    averaged_scores = [
        {'Residue': residue, f'{fragment_size}aa_Avg_Score': data['sum'] / data['count']}
        for residue, data in residue_scores.items()
    ]

    # Convert to DataFrame
    return pd.DataFrame(averaged_scores)

# List of fragment sizes and corresponding CSV files
fragment_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]
csv_files = [f'tau_{size}aa_peptides.csv' for size in fragment_sizes]

# Compute per-residue scores for each fragment size
all_per_residue_scores = []
for size, csv_file in zip(fragment_sizes, csv_files):
    per_residue_scores = compute_per_residue_scores(csv_file, size)
    all_per_residue_scores.append(per_residue_scores)

# Merge all per-residue scores into a single DataFrame
merged_scores = all_per_residue_scores[0]
for additional_scores in all_per_residue_scores[1:]:
    merged_scores = pd.merge(merged_scores, additional_scores, on='Residue', how='outer')

# Save the merged scores to a pickled file and CSV
output_csv = 'combined_per_res_scores.csv'
merged_scores.to_csv(output_csv, index=False)

# Indicate completion and file locations
output_pickle, output_csv
