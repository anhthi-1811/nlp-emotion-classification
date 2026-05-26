# ==========================================
# FILE: preprocessor.py
# ROLE: Object-Oriented Text Preprocessing Pipeline
# ==========================================
import re
import pandas as pd
from typing import Union, List
from tqdm import tqdm 

class DLTextPreprocessor:
    """
    A professional-grade text preprocessor optimized for Deep Learning models (BERT/LSTM).
    Preserves semantic context, handles contractions, and removes digital noise.
    """
    
    def __init__(self):
        # Encapsulate the contraction map inside the class so it doesn't pollute the global scope
        self.CONTRACTION_MAP = {
            "i'm": "i am", "can't": "cannot", "won't": "will not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "it's": "it is", "that's": "that is"
        }
        
    def _clean_single_text(self, text: str) -> str:
        """
        Internal method (hidden) to process a single string.
        """
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in self.CONTRACTION_MAP.items():
            text = re.sub(r"\b" + contraction + r"\b", expansion, text)
            
        # Remove digital noise (URLs, mentions, HTML)
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'&\w+;|<[^>]+>', ' ', text)
        
        # Keep only alphabets and expressive punctuations
        text = re.sub(r'[^a-z\s\!\?\.\,]', ' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def transform(self, data: Union[str, List[str], pd.Series]) -> Union[str, List[str]]:
        """
        Main method intelligently handles single strings, Lists, or Pandas Series.
        Auto-triggers a progress bar for large datasets.
        """
        if isinstance(data, str):
            return self._clean_single_text(data)
            
        elif isinstance(data, (list, pd.Series)):
            # Tự động kích hoạt thanh tiến trình nếu dữ liệu lớn hơn 1000 dòng
            if len(data) > 1000:
                from tqdm import tqdm
                return [self._clean_single_text(text) for text in tqdm(data, desc="Purifying Text")]
            else:
                return [self._clean_single_text(text) for text in data] 
            
        else:
            raise TypeError("Input must be a string, list of strings, or pandas Series.")