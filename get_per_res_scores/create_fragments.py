#%%
import numpy as np
# Function to parse a FASTA file and extract the sequence
def parse_fasta(file_path):
    with open(file_path, 'r') as fasta_file:
        lines = fasta_file.readlines()
    header = lines[0].strip()  # Extract the header
    sequence = ''.join(line.strip() for line in lines[1:])  # Combine sequence lines
    return header, sequence

# Function to generate peptides and write them to a FASTA file
def generate_peptides(header, sequence, fragment_size, sliding_window, output_file):
    with open(output_file, 'w') as fasta_out:
        for start in np.arange(0,len(sequence) - fragment_size + 1, sliding_window):
            peptide = sequence[start:start + fragment_size]
            fasta_out.write(f">{header}_{fragment_size}aa_{start+1}-{start+fragment_size}\n")
            fasta_out.write(f"{peptide}\n")

# Main script to create peptide FASTA files for sizes 10aa to 50aa
def process_fasta(input_fasta):
    header, sequence = parse_fasta(input_fasta)
    header = header[1:]  # Remove the ">" character
    for size in np.arange(10, 51, 5):  # Generate for sizes 10aa to 50aa
        output_file = f"{header}_{size}aa_peptides.fasta"
        generate_peptides(header, sequence, size, 5, output_file)
    return "Peptide FASTA files generated successfully."

# Run the script with the uploaded FASTA file
result_message = process_fasta('tau.fasta')
result_message
