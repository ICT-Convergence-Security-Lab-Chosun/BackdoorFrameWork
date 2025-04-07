import os
import pandas as pd
from datasets import load_dataset



def save_data(dataset, path, split):
    if path is not None:
        os.makedirs(path, exist_ok=True)
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(os.path.join(path, f'{split}.csv'))
        
if __name__=='__main__':
    # Define the output directory and create it if it doesn't exist
    output_dir = "./datasets/SentimentAnalysis/SST-2/client"
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.tsv")
    dev_file = os.path.join(output_dir, "dev.tsv")
    test_file = os.path.join(output_dir, "test.tsv")

    #Check if the directory already contains data
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

    #Check if the directory already contains data
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
    

