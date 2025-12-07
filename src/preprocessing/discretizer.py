import pandas as pd
from math import log2
from typing import Tuple, Dict

class Discretizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.class_col = self.df.iloc[:, -1]
        self.dim_cols = self.df.iloc[:, :-1]
        self.results = None
        
    def set_input_path(self, new_path: str):
        self.df = pd.read_csv(filepath_or_buffer=new_path, sep=',', header=None, index_col=False)
        
        self.class_col = self.df.iloc[:, -1]
        self.dim_cols = self.df.iloc[:, :-1]
    
    def compute_entropy(self, data_classes: pd.Series) -> float:
        # Obtiene la frecuencia de cada clase y el total de muestras
        classes = data_classes.value_counts(sort=False)
        n = classes.sum()
        entropy = 0.0
        
        for c in classes:
            proportion = c / n
            if proportion > 0:
                entropy -= proportion * log2(proportion)
        
        return entropy
    
    def get_quartiles(self, col: pd.Series) -> pd.Series:
        return col.quantile([0.25, 0.5, 0.75])
    
    def split_by_cutpoint(self, col: pd.Series, cutpoint: float) -> Tuple[pd.Series, pd.Series]:
        # Divide las clases según el punto de corte
        left = self.class_col[col < cutpoint]
        right = self.class_col[col >= cutpoint]
        return left, right
    
    def compute_split_gain(self, left: pd.Series, right: pd.Series) -> float:
        N = len(self.class_col)
        N_left = len(left)
        N_right = len(right)
        
        if N_left == 0 or N_right == 0:
            return -1
        
        entropy_full = self.compute_entropy(self.class_col)
        entropy_left = self.compute_entropy(left)
        entropy_right = self.compute_entropy(right)
        
        post_split_entropy = (N_left / N) * entropy_left + (N_right / N) * entropy_right

        return entropy_full - post_split_entropy
    
    def evaluate_column(self, col_index: str) -> Tuple[float, float, int]:
        col = self.dim_cols[col_index]
        quartiles = self.get_quartiles(col)
        
        best_gain_col = -1
        best_split_col = None
        quartile = 0
        
        for i, q in enumerate(quartiles):
            left, right = self.split_by_cutpoint(col, q)
            gain = self.compute_split_gain(left, right)
            
            if gain > best_gain_col:
                best_gain_col = gain
                best_split_col = q
                quartile = i + 1
        
        return best_gain_col, best_split_col, quartile
    
    def discretize(self) -> Tuple[pd.DataFrame, Dict]:
        self.results = []
        
        for col_index in self.dim_cols.columns:
            gain, split, quartile = self.evaluate_column(col_index)
            
            if split is None:
                self.df[col_index] = 0
                
                results = {
                    'Columna' : col_index + 1,
                    'Cuartil usado' : 'NA',
                    'Punto de corte' : 'NA',
                    'Ganancia de información' : 0.0
                }
                self.results.append(results)
                
            else:
                filter = self.df[col_index] < split
                self.df.loc[filter, col_index] = 0
                self.df.loc[~filter, col_index] = 1
                
                results = {
                    'Columna' : col_index + 1,
                    'Cuartil usado' : f"Q{quartile}",
                    'Punto de corte': split,
                    'Ganancia de información' : gain
                }
                
                self.results.append(results)
                
        return self.df, self.results