import numpy as np
import pandas as pd

class Normalizer:
    def min_max(self, df: pd.DataFrame, lower_bound: float, upper_bound: float) -> pd.DataFrame:
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        min_vals = dims.min(axis=0)
        max_vals = dims.max(axis=0)
        
        normalized_df = (dims- min_vals) / (max_vals - min_vals) * (upper_bound - lower_bound) + lower_bound

        return pd.concat([normalized_df, classes], axis=1)
    
    def z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        mean_vals = dims.mean(axis=0)
        std_vals = dims.std(axis=0)
        
        normalized_df = (dims - mean_vals) / std_vals
        
        return pd.concat([normalized_df, classes], axis=1)
    
    def decimal_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        max_abs_vals = dims.abs().max(axis=0)
        j_vals = max_abs_vals.apply(lambda x: len(str(int(np.floor(x)))) if x >= 1 else 0)
        
        normalized_df = df / (10 ** j_vals)
        
        return pd.concat([normalized_df, classes], axis=1)