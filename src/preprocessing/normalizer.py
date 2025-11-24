"""
Módulo que implementa los algoritmos de normalización a través
de la clase Normalizer.
"""

import numpy as np
import pandas as pd

class Normalizer:
    """
    Clase que implementa los algoritmos de normalización solicitados.
    """
    
    def min_max(
        self, 
        df: pd.DataFrame, 
        lower_bound: float, 
        upper_bound: float
    ) -> pd.DataFrame:
        """
        Aplica la normalización Min-Max al DataFrame dado.
        """
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        # Calcula los valores mínimos y máximos
        min_vals = dims.min(axis=0)
        max_vals = dims.max(axis=0)
        
        # Aplica la normalización
        normalized_df = (dims- min_vals) / (max_vals - min_vals) * (upper_bound - lower_bound) + lower_bound

        # Concatena la columna de las clases
        return pd.concat([normalized_df, classes], axis=1)
    
    def z_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la normalización Z-Score al DataFrame dado.
        """
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        # Calcula la media y desviación estándar
        mean_vals = dims.mean(axis=0)
        std_vals = dims.std(axis=0)
        
        # Aplica la normalización
        normalized_df = (dims - mean_vals) / std_vals
        
        # Concatena la columna de las clases
        return pd.concat([normalized_df, classes], axis=1)
    
    def decimal_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la normalización por escalado decimal al DataFrame dado.
        """
        # Divide el dataframe en dimensiones y clases
        dims = df.iloc[:, :-1]
        classes = df.iloc[:, -1]
        
        # Obtiene el valor absoluto máximo de cada dimensión
        max_abs_vals = dims.abs().max(axis=0)
        # Calcula j para cada columna (número de dígitos)
        j_vals = max_abs_vals.apply(lambda x: len(str(int(np.floor(x)))) if x >= 1 else 0)
        
        # Aplica la normalización
        normalized_df = df / (10 ** j_vals)
        
        # Concatena la columna de las clases
        return pd.concat([normalized_df, classes], axis=1)