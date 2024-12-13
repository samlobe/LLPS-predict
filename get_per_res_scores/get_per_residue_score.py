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
# fragment_sizes = [50]
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
output_csv = f'tau_per_res_scores.csv'
merged_scores.to_csv(output_csv, index=False)

# Indicate completion and file locations
output_csv

#%%
import matplotlib.pyplot as plt
# plot the per-residue for each fragment length on the same plot
plt.figure(figsize=(12, 6))

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

import matplotlib.pyplot as plt
from itertools import cycle

# Use a pre-defined color palette from matplotlib
colors = cycle(plt.cm.tab10.colors)  # Tableau 10 Palette (default in Matplotlib)
colors = cycle(plt.cm.viridis.colors[::30])  # Viridis Palette (colorblind-friendly)

# Plot each size with a unique color
for size in fragment_sizes:
    color = next(colors)  # Get the next color from the cycle
    plt.plot(
        merged_scores['Residue'], 
        merged_scores[f'{size}aa_Avg_Score'], 
        label=f'{size}aa', 
        color=color
    )

plt.xlabel('Residue', fontsize=14)
plt.ylabel('Average LLPS Score', fontsize=14)
# plt.legend(fontsize=12)

# label the legend as fragment probe length
plt.legend(title='Fragment Probe Length', fontsize=14, ncol=2, title_fontsize=14)

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylim(0, 1)
plt.xlim(0, 441)
plt.title("tau's LLPS propensity per residue", fontsize=16)

# shade from 1–165 and label it "N-terminal domain" with text
plt.axvspan(1, 165, color='gray', alpha=0.2)
plt.text(85, 0.1, 'N-terminal domain', fontsize=12, ha='center')
# shade from 166-242 and label it "Proline-rich region" with text
plt.axvspan(166, 242, color='red', alpha=0.2)
plt.text(204, 0.1, 'Proline-rich region', fontsize=12, ha='center')
# shade 242-274 (136, 204, 238) and label it "R1" with text
plt.axvspan(242, 274, color=[136/256,204/256,238/256], alpha=0.3)
plt.text(258, 0.1, 'R1', fontsize=12, ha='center')
# shade 275-305 (204, 102, 85) and label it "R2" with text
plt.axvspan(275, 305, color=[204/256,102/256,85/256], alpha=0.3)
plt.text(290, 0.1, 'R2', fontsize=12, ha='center')
# shade 306-337 (221, 204, 119) and label it "R3" with text
plt.axvspan(306, 337, color=[221/256,204/256,119/256], alpha=0.3)
plt.text(321, 0.1, 'R3', fontsize=12, ha='center')
# shade 338-378 (136, 34, 85) and label it "R4" with text
plt.axvspan(338, 378, color=[136/256,34/256,85/256], alpha=0.3)
plt.text(358, 0.1, 'R4', fontsize=12, ha='center')
# shade 379-441 (gray and label it "C-terminal domain" with text
plt.axvspan(379, 441, color='gray', alpha=0.3)
plt.text(410, 0.1, 'C-terminal domain', fontsize=12, ha='center')