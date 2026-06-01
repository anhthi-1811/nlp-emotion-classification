# ==========================================
# FILE: preprocessor.py
# ROLE: Object-Oriented Text Preprocessing Pipeline
# ==========================================
import re
import pandas as pd
from typing import Union, List
from tqdm import tqdm 
import contractions
from emot.core import emot

class DLTextPreprocessor:
    """
    A professional-grade text preprocessor optimized for Deep Learning models.
    Auto-translates emoticons, handles GoEmotions tags, expands contractions,
    and removes digital noise while preserving semantic context.
    """
    
    def __init__(self):
        # Initialize the auto-emoticon translation engine once to save memory/time
        self.emot_engine = emot()
        
    def _clean_single_text(self, text: str) -> str:
        """
        Internal method (hidden) to process a single string.
        """
        if not isinstance(text, str):
            text = str(text)
            
        # 1. REMOVE DIGITAL NOISE FIRST! (CRITICAL FIX)
        # Must remove URLs before translating emoticons, otherwise ':/' in 'http://' 
        # gets translated into 'confused face' and the URL is permanently corrupted.
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'&\w+;|<[^>]+>', ' ', text)
        
        # 2. Handle GoEmotions specific tags
        text = re.sub(r'\[NAME\]', 'someone', text)
        text = re.sub(r'\[RELIGION\]', 'religion', text)
        text = re.sub(r'\[[A-Z]+\]', ' ', text)
        
        # 3. EMOTION PRESERVATION (AUTOMATED)
        # Safe to translate now since URLs are purged.
        emoticon_info = self.emot_engine.emoticons(text)
        if emoticon_info and isinstance(emoticon_info, dict) and emoticon_info.get('flag'):
            for value, mean in zip(emoticon_info['value'], emoticon_info['mean']):
                clean_mean = re.sub(r'[^a-zA-Z\s]', '', mean).lower()
                text = text.replace(value, f" {clean_mean} ")
        
        # 4. Expand ALL English contractions automatically
        text = contractions.fix(text)
        
        # 5. Convert to lowercase 
        text = text.lower()
                
        # 6. Strict Regex: Keep ONLY alphabets, spaces, and basic expressive punctuations
        text = re.sub(r'[^a-z\s\!\?\.\,]', ' ', text)
        
        # 7. Collapse multiple consecutive spaces into a single space and strip edges
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def transform(self, data: Union[str, List[str], pd.Series]) -> Union[str, List[str], pd.Series]:
        """
        Main method intelligently handles single strings, Lists, or Pandas Series.
        Auto-triggers a progress bar for large datasets.
        """
        if isinstance(data, str):
            return self._clean_single_text(data)
            
        elif isinstance(data, pd.Series):
            # Native Pandas application to strictly preserve Index alignment (CRITICAL FIX)
            if len(data) > 1000:
                tqdm.pandas(desc="Purifying Text Series")
                return data.progress_apply(self._clean_single_text)
            else:
                return data.apply(self._clean_single_text)
                
        elif isinstance(data, list):
            if len(data) > 1000:
                return [self._clean_single_text(text) for text in tqdm(data, desc="Purifying Text List")]
            else:
                return [self._clean_single_text(text) for text in data] 
            
        else:
            raise TypeError("Input must be a string, list of strings, or pandas Series.") 