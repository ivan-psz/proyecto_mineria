import pandas as pd
from math import log2
from typing import Tuple, Dict

class Discretizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.class_col = df.iloc[:, -1]
        self.dim_cols = df.iloc[:, :-1]
        self.best_col = None
        self.best_gain = -1
        self.best_split = None
        
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
    
    def evaluate_column(self, col_index: str) -> Tuple[float, float]:
        # Obtiene la columna de la dimensión y sus cuartiles
        col = self.dim_cols[col_index]
        quartiles = self.get_quartiles(col)
        
        best_gain_col = -1
        best_split_col = None
        
        for q in quartiles:
            # Por cada cuartil, divide y calcula la ganancia
            left, right = self.split_by_cutpoint(col, q)
            gain = self.compute_split_gain(left, right)
            
            if gain > best_gain_col:
                best_gain_col = gain
                best_split_col = q
        
        return best_gain_col, best_split_col
    
    def compute_info_gain(self):
        best_col = None
        best_gain = -1
        best_split = None
        
        for col_index in self.dim_cols.columns:
            gain, split = self.evaluate_column(col_index)
            
            if gain > best_gain:
                best_gain = gain
                best_col = col_index
                best_split = split
                
        self.best_col = best_col
        self.best_gain = best_gain
        self.best_split = best_split
        
    def discretize(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        # Divide el DataFrame según el mejor punto de corte
        filter = self.df[self.best_col] < self.best_split
        df_left = self.df[filter]
        df_right = self.df[~filter]
        
        metadata = {
            'Índice_mejor_columna': [self.best_col],
            'Mejor_punto_corte': [self.best_split],
            'Ganancia_información': [self.best_gain]
        }
        
        return df_left, df_right, metadata