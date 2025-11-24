"""
Módulo que implementa la discretización basada en entropía a través
de la clase Discretizer.
"""

import pandas as pd
from math import log2
from typing import Tuple, Dict

class Discretizer:
    """
    Clase para discretizar atributos numéricos en dos particiones
    basados en la ganancia de información usando cuartiles como puntos de corte.
    
    Atributos:
        df (pd.DataFrame): DataFrame original.
        class_col (pd.Series): Columna de clase (última columna del DataFrame).
        dim_cols (pd.DataFrame): Columnas de dimensiones.
        best_col (str): Índice de la mejor columna para discretizar.
        best_gain (float): Máxima ganancia de información encontrada.
        best_split (float): Punto de corte óptimo para la mejor partición.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el discretizador con el DataFrame dado.
        """
        self.df = df
        self.class_col = df.iloc[:, -1]     # Última columna
        self.dim_cols = df.iloc[:, :-1]     # Todas menos la última
        self.best_col = None
        self.best_gain = -1
        self.best_split = None
        
    def set_input_path(self, new_path: str):
        """
        Cambia el archivo de entrada y recarga los datos en el DataFrame.
        """
        self.df = pd.read_csv(filepath_or_buffer=new_path, sep=',', header=None, index_col=False)
        
        # Actualiza las columnas de clase y dimensiones
        self.class_col = self.df.iloc[:, -1]
        self.dim_cols = self.df.iloc[:, :-1]
    
    def compute_entropy(self, data_classes: pd.Series) -> float:
        """
        Calcula la entropía de un conjunto de clases.
        
        Parámetros:
            data_classes (pd.Series): Serie con las clases.
        """
        # Obtiene la frecuencia de cada clase y el total de muestras
        classes = data_classes.value_counts(sort=False)
        n = classes.sum()
        entropy = 0.0
        
        for c in classes:
            # Calcula la proporción de cada clase
            proportion = c / n
            if proportion > 0:
                entropy -= proportion * log2(proportion)
        
        return entropy
    
    def get_quartiles(self, col: pd.Series) -> pd.Series:
        """
        Obtiene los cuartiles Q1, Q2 y Q3 de una columna dada.
        
        Parámetros:
            col (pd.Series): Columna de la dimensión.
        """
        return col.quantile([0.25, 0.5, 0.75])
    
    def split_by_cutpoint(self, col: pd.Series, cutpoint: float) -> Tuple[pd.Series, pd.Series]:
        """
        Divide la columna de la clase en dos grupos según un punto de corte sobre una columna de dimensión.
        
        Parámetros:
            col (pd.Series): Columna de la dimensión.
            cutpoint (float): Punto de corte.
        
        Devuelve:
            Tuple[pd.Series, pd.Series]: Clases a la izquierda y derecha del corte.
        """
        # Divide las clases según el punto de corte
        left = self.class_col[col < cutpoint]
        right = self.class_col[col >= cutpoint]
        return left, right
    
    def compute_split_gain(self, left: pd.Series, right: pd.Series) -> float:
        """
        Calcula la ganancia de información al dividir las clases en dos grupos.
        
        Parámetros:
            left (pd.Series): Clases del grupo izquierdo.
            right (pd.Series): Clases del grupo derecho.
        """
        # Número total de muestras y de cada partición
        N = len(self.class_col)
        N_left = len(left)
        N_right = len(right)
        
        if N_left == 0 or N_right == 0:
            # Si alguna partición queda vacía, la ganancia no es válida
            return -1
        
        # Calcula las entropías necesarias
        entropy_full = self.compute_entropy(self.class_col)
        entropy_left = self.compute_entropy(left)
        entropy_right = self.compute_entropy(right)
        
        # Calcular la entropía posterior a la partición.
        post_split_entropy = (N_left / N) * entropy_left + (N_right / N) * entropy_right

        return entropy_full - post_split_entropy
    
    def evaluate_column(self, col_index: str) -> Tuple[float, float]:
        """
        Evalúa todas las particiones posibles de una columna y devuelve la de mayor ganancia de información y su punto de corte.
        
        Parámetros:
            col_index (str): Índice de la columna a evaluar.
        """
        # Obtiene la columna de la dimensión y sus cuartiles
        col = self.dim_cols[col_index]
        quartiles = self.get_quartiles(col)
        
        best_gain_col = -1
        best_split_col = None
        
        for q in quartiles:
            # Por cada cuartil, divide y calcula la ganancia
            left, right = self.split_by_cutpoint(col, q)
            gain = self.compute_split_gain(left, right)
            
            # Si la ganancia es mejor, actualiza los valores
            if gain > best_gain_col:
                best_gain_col = gain
                best_split_col = q
        
        return best_gain_col, best_split_col
    
    def compute_info_gain(self):
        """
        Busca la mejor columna y punto de corte en todo el DataFrame,
        evaluando los cuartiles de cada columna.
        """
        best_col = None
        best_gain = -1
        best_split = None
        
        for col_index in self.dim_cols.columns:
            # Evalua la columna actual
            gain, split = self.evaluate_column(col_index)
            
            # Si la ganancia es mejor, actualizar los valores globales
            if gain > best_gain:
                best_gain = gain
                best_col = col_index
                best_split = split
                
        self.best_col = best_col
        self.best_gain = best_gain
        self.best_split = best_split
        
    def discretize(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Discretiza el DataFrame en dos particiones usando la mejor columna y punto de corte encontrados.
        
        Devuelve:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: 
                DataFrames de la partición izquierda, derecha y metadatos.
        """
        # Divide el DataFrame según el mejor punto de corte
        left_mask = self.df[self.best_col] < self.best_split
        df_left = self.df[left_mask]
        df_right = self.df[~left_mask]
        
        # Guarda los metadatos en listas para poder guardarlos en CSV
        metadata = {
            'Índice_mejor_columna': [self.best_col],
            'Mejor_punto_corte': [self.best_split],
            'Ganancia_información': [self.best_gain]
        }
        
        return df_left, df_right, metadata