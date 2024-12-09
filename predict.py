# LLPS_predictor.py
import torch
import esm
import argparse
import joblib
import pandas as pd
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description='Predict LLPS propensity of a peptide or protein (designed for IDRs).')
parser.add_argument('--sequence', '-s', help='Protein sequence (or a .fasta file) for which embeddings will be generated. Use this if you do not have embeddings.')
parser.add_argument('--embeddingsDir', help='Directory containing .pt embeddings files for protein sequences.')
parser.add_argument('--embeddingsFile', help='Single .pt file containing embeddings for a sequence.')
parser.add_argument('--LR_model', help='Logistic Regression model file (.joblib) for LLPS prediction.', default='model_development/LLPS_model_latest.joblib')
parser.add_argument('--output', '-o', help='Output CSV file for predictions. Default is LLPS_propensity.csv', default='LLPS_propensity.csv')
parser.add_argument('--nogpu', action='store_true', help='Disable GPU usage if you encounter memory issues.')
parser.add_argument('--ESM_model', help='ESM model to use for generating embeddings. Options: 3B (may add other models later)', default='3B')
parser.add_argument('--embeddings_output', help='If you want to save the generated embeddings, provide a file name. Will save as .npy')

args = parser.parse_args()

# Function to load ESM model
def load_esm_model(model_size):
    if model_size == '3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        layer = 36
    # elif model_size == '15B': # don't yet support 15B
    #     model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    #     layer = 48
    else:
        raise ValueError(f"Unsupported ESM model: {model_size}")
    return model, alphabet, layer

# Load and process embeddings from .fasta or a sequence
def get_embeddings_from_sequence(sequence, esm_model, alphabet, layer=36, nogpu=False):
    if isinstance(sequence, str) and (sequence.endswith('.fasta') or sequence.endswith('.fa')):
        # Load sequences from fasta
        with open(sequence) as f:
            lines = f.readlines()
        names, sequences = [], []
        for line in lines:
            if line.startswith('>'):
                names.append(line[1:].strip())
            else:
                sequences.append(line.strip())
        if len(names) != len(sequences):
            raise ValueError("Mismatch between number of sequence names and sequences in the .fasta file.")
    else:
        names, sequences = [sequence], [sequence]

    batch_converter = alphabet.get_batch_converter()
    data = [(name, seq) for name, seq in zip(names, sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if torch.cuda.is_available() and not nogpu:
        esm_model = esm_model.cuda()
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[layer], return_contacts=False)
    
    embeddings = [results['representations'][layer][i, 1:len(seq) + 1].mean(0) for i, seq in enumerate(sequences)]
    return torch.stack(embeddings).cpu().numpy(), names

# Load embeddings from a .pt file
def load_embeddings_from_pt_file(pt_file, layer):
    data = torch.load(pt_file, weights_only=True)
    if 'mean_representations' in data and layer in data['mean_representations']:
        embeddings = data['mean_representations'][layer].numpy()
        return embeddings[np.newaxis, :], os.path.splitext(os.path.basename(pt_file))[0]
    else:
        raise ValueError(f"The .pt file {pt_file} does not contain the required 'mean_representations' or layer {layer}.")

# Main function to process input and generate predictions
def main():
    if args.ESM_model not in ['3B']:
        raise ValueError("Invalid ESM model specified. Only '3B' is supported currently.")
    
    # Load ESM model if generating embeddings from sequence
    if args.sequence:
        esm_model, alphabet, layer = load_esm_model(args.ESM_model)
        embeddings, names = get_embeddings_from_sequence(args.sequence, esm_model, alphabet, layer=layer, nogpu=args.nogpu)
        if args.embeddings_output:
            np.save(args.embeddings_output, embeddings)
            print(f"Embeddings saved to {args.embeddings_output}")
    
    # Load embeddings from .pt file or directory
    elif args.embeddingsFile:
        esm_model, alphabet, layer = load_esm_model(args.ESM_model)
        embeddings, names = load_embeddings_from_pt_file(args.embeddingsFile, layer)

    elif args.embeddingsDir:
        esm_model, alphabet, layer = load_esm_model(args.ESM_model)
        embeddings = []
        names = []
        for file in os.listdir(args.embeddingsDir):
            if file.endswith('.pt'):
                emb, name = load_embeddings_from_pt_file(os.path.join(args.embeddingsDir, file), layer)
                embeddings.append(emb)
                names.append(name)
        embeddings = np.vstack(embeddings)
    
    # Load Logistic Regression model
    loaded_data = joblib.load(args.LR_model)
    LR_model = loaded_data['model']
    scaler = loaded_data['scaler']
    feature_indices = loaded_data['feature_indices']
    print("Logistic Regression model loaded successfully.")

    # scale embeddings
    embeddings_scaled = scaler.transform(embeddings)
    # select features
    embeddings_selected = embeddings_scaled[:, feature_indices]

    # Predict LLPS propensity
    predictions = LR_model.predict_proba(embeddings_selected)[:, 1]
    if len(names) == 1:
	    print(f"LLPS probabilities: {predictions[0]:.4f}")

    # Save predictions to CSV
    results_df = pd.DataFrame({'Name': names, 'LLPS Score': predictions})
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Plot if more than one sequence
    if len(names) > 1 and len(names) < 100:
        plt.bar(names, predictions)
        plt.xlabel('Protein')
        plt.ylabel('LLPS Score')
        plt.xticks(rotation=90)
        plt.title(f'LLPS Propensity Predictions - ESM {args.ESM_model}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
