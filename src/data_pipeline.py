# src/data_pipeline.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    """Custom PyTorch Dataset for loading multi-label emotion text and targets."""
    def __init__(self, dataframe):
        # Reset index to avoid KeyError when iterating
        self.dataframe = dataframe.reset_index(drop=True)
        self.texts = self.dataframe['text'].values
        
        # Dynamically extract all 28 emotion label columns (ignoring 'text' and 'id')
        self.label_cols = [col for col in self.dataframe.columns if col not in ['text', 'id']]
        self.labels = self.dataframe[self.label_cols].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_array = self.labels[idx]
        
        label_tensor = torch.tensor(label_array, dtype=torch.float32)
        
        return text, label_tensor


class DataPipelineManager:
    """Manages the entire data engineering workflow for Task 1."""
    def __init__(self, raw_data_path, mock_data_output_path):
        self.raw_data_path = raw_data_path
        self.mock_data_output_path = mock_data_output_path
        self.df_raw = None

    def load_raw_data(self):
        """Loads the heavy raw dataset (supports both Parquet and CSV)."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")
        
        # Branching logic to handle the new GoEmotions Parquet format
        if self.raw_data_path.endswith('.parquet'):
            self.df_raw = pd.read_parquet(self.raw_data_path)
        else:
            self.df_raw = pd.read_csv(self.raw_data_path)
            
        return self.df_raw

    def generate_mock_data(self, num_samples=5000, random_state=42):
        """Extracts and saves a smaller split for the modeling team prototyping."""
        if self.df_raw is None:
            self.load_raw_data()
        
        # Ensure we filter out unnecessary metadata from the original Parquet
        # and only keep the text and the 28 emotion columns
        target_cols = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse",
            "sadness", "surprise", "neutral"
        ]
        
        # Keep 'text' + the 28 emotion columns if they exist in the dataframe
        cols_to_keep = ['text'] + [col for col in target_cols if col in self.df_raw.columns]
        
        df_mock = self.df_raw[cols_to_keep].sample(n=num_samples, random_state=random_state)
        
        os.makedirs(os.path.dirname(self.mock_data_output_path), exist_ok=True)
        df_mock.to_csv(self.mock_data_output_path, index=False)
        return df_mock

    def get_pytorch_loaders(self, df, batch_size=32, shuffle=True):
        """Wraps a dataframe into an executable PyTorch DataLoader."""
        dataset = EmotionDataset(df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader 