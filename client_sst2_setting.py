import csv
import os
import pandas as pd
from datasets import load_dataset
import numpy as np
import random

def save_data(dataset, path, split):
    if path is not None:
        os.makedirs(path, exist_ok=True)
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(os.path.join(path, f'{split}.csv'))

# Function to save non-IID data
def create_noniid_data(df, n_clients=100, n_samples_per_client=6000, alpha=0.5, random_seed=42):
    """
    df: source data (train.tsv)
    n_clients: Number of clients to create (for example, 100)
    n_samples_per_client: number of samples per client (e.g. 6000)
    alpha: Beta distribution parameters (smaller, one-sided distribution)
    random_seed: seed value for reproduction
    """
    np.random.seed(random_seed)
    
    # Split data by label (assuming binary classification: 0 and 1)
    df0 = df[df["1"] == 0]
    df1 = df[df["1"] == 1]
    
    data0 = df0.to_dict(orient="records")
    data1 = df1.to_dict(orient="records")
    
    print(f"Total label 0 samples: {len(data0)}, label 1 samples: {len(data1)}")
    
    clients = {}
    
    for client_id in range(1, n_clients+1):
        # Determine the proportion of label 0 using Beta distribution (Beta(alpha, alpha))
        ratio_0 = np.random.beta(alpha, alpha)
        n0 = int(round(ratio_0 * n_samples_per_client))
        n1 = n_samples_per_client - n0
        
        # Sample data for each label (allow duplicates if data is insufficient)
        samples0 = np.random.choice(data0, size=n0, replace=True).tolist() if n0 > 0 else []
        samples1 = np.random.choice(data1, size=n1, replace=True).tolist() if n1 > 0 else []
        client_samples = samples0 + samples1
        
        # Shuffle the entire dataset
        np.random.shuffle(client_samples)
        
        # Create DataFrame
        client_df = pd.DataFrame(client_samples)
        # To match the final file format, additional columns (e.g., always 0) will be added
        clients[client_id] = client_df
        
        print(f"Client {client_id}: {n0} samples of label 0, {n1} samples of label 1")
    
    return clients
        
if __name__=='__main__':
    # Define the output directory and create it if it doesn't exist
    output_dir = "./datasets/SentimentAnalysis/SST-2/Federated_Learning"
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.tsv")
    dev_file = os.path.join(output_dir, "dev.tsv")
    test_file = os.path.join(output_dir, "test.tsv")

    # Check if the directory already contains data
    if os.path.exists(train_file) and os.path.exists(dev_file) and os.path.exists(test_file):
        print("Data already exists in the specified directory.")
    else:
        # Load the training dataset from stanfordnlp/sst2
        train_dataset = load_dataset("stanfordnlp/sst2", split="train")
        df_train = pd.DataFrame(train_dataset)[["sentence", "label"]]

        # Load the validation and test datasets from SetFit/sst2
        setfit_dataset = load_dataset("SetFit/sst2")
        df_dev = pd.DataFrame(setfit_dataset["validation"])[["text", "label"]]
        df_test = pd.DataFrame(setfit_dataset["test"])[["text", "label"]]

        # If you want to keep the same column names across files, you can rename them:
        df_dev = df_dev.rename(columns={"text": "sentence"})
        df_test = df_test.rename(columns={"text": "sentence"})

        # Define output file paths
        train_file = os.path.join(output_dir, "train.tsv")
        dev_file = os.path.join(output_dir, "dev.tsv")
        test_file = os.path.join(output_dir, "test.tsv")

        # Save the DataFrames as TSV files
        df_train.to_csv(train_file, sep="\t", index=False)
        df_dev.to_csv(dev_file, sep="\t", index=False)
        df_test.to_csv(test_file, sep="\t", index=False)

        print(f"Train data saved to {train_file}")
        print(f"Validation data saved to {dev_file}")
        print(f"Test data saved to {test_file}")
    
    # Check if the directory already contains train.csv
    train_file = os.path.join(output_dir, "train.csv")
    dev_file = os.path.join(output_dir, "dev.csv")
    test_file = os.path.join(output_dir, "test.csv")

    # Check if the directory already contains data
    if os.path.exists(train_file) and os.path.exists(dev_file) and os.path.exists(test_file):
        print("Data already exists in the specified directory.")
    else:
        # Load the dataset from clientsst-2
        from openbackdoor import load_dataset
        target_dataset = load_dataset(name="clientsst-2")
        # Save the datasets in the specified format
        save_data(target_dataset["train"], output_dir, "train-clean")
        save_data(target_dataset["dev"], output_dir, "dev-clean")
        save_data(target_dataset["test"], output_dir, "test-clean")

    train_clean_path = os.path.join(output_dir, "train-clean.csv")
    if os.path.exists(os.path.join(output_dir, "client/100/noniid-train.csv")):
        print("Non IID Data already exists in the specified directory.")
    else:
        df_train_clean = pd.read_csv(train_clean_path)
        print(f"Loaded train-clean.csv with {len(df_train_clean)} records.")
        # Generate non-IID data: 100 clients, 6000 samples each
        clients_data = create_noniid_data(df_train_clean, n_clients=100, n_samples_per_client=6000, alpha=0.5)
        # Create a folder for each client and save noniid-train.csv
        for client_id, client_df in clients_data.items():
            # Path for the client folder (from 1 to 100)
            client_folder = os.path.join(output_dir, "client", str(client_id))
            os.makedirs(client_folder, exist_ok=True)
            # Add additional columns to match the final file format:
            # 1) The DataFrame currently has "sentence" and "label" columns.
            # 2) Add a final column with all values set to 0.
            # 3) Save the file including the index as the first column, and save the header.
            final_df = client_df[["0", "1"]].copy()
            final_df["2"] = 0   # Add the final column, all values set to 0
            
            # Reset the index to start from 0
            final_df.reset_index(drop=True, inplace=True)
            
            # Save: Include the index so that the first column becomes the index.
            output_file = os.path.join(client_folder, "noniid-train.csv")
            final_df.to_csv(output_file, index=True, header=True, quoting=csv.QUOTE_MINIMAL)
            
            print(f"Client {client_id} noniid-train.csv saved to {output_file}")