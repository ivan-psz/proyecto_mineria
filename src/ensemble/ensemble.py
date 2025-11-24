"""
Módulo que implementa el Ensemble del Bienestar, un conjunto de árboles de decisión
que utilizan muestreo aleatorio de instancias y atributos.

Arriba la 4T.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .validator import Validator
from datetime import datetime

class EnsembleDelBienestar:
    """
    Clase que implementa el ensemble con un conjunto de árboles de decisión
    que utilizan muestreo aleatorio de instancias y atributos.
    
    Atributos:
        output_path (str): Ruta del directorio donde se guardarán los resultados.
        n_trees (int): Número de árboles en el ensemble.
        sample_with_replacement (bool): Si se debe muestrear con reemplazo las instancias.
        attribute_with_replacement (bool): Si se debe muestrear con reemplazo los atributos.
        random_state (int): Semilla para reproducibilidad.
        trees (List[]): Lista de árboles en el ensemble.
        attribute_indices (List[]): Lista de índices de atributos usados por cada árbol.
        validator (Validator): Instancia de la clase Validator para validación cruzada.
    """
    def __init__(
        self,
        output_path: str,
        n_trees: int,
        sample_with_replacement: bool,
        attribute_with_replacement: bool,
        random_state: int = 42
    ):
        """
        Inicializa el Ensemble del Bienestar con los parámetros dados.
        
        Parámetros:
            output_path (str): Ruta del directorio de salida.
            n_trees (int): Número de árboles en el ensemble.
            sample_with_replacement (bool): Si se debe muestrear con reemplazo las muestras.
            attribute_with_replacement (bool): Si se debe muestrear con reemplazo los atributos.
            random_state (int): Semilla
        """
        self.output_path = output_path
        self.n_trees = n_trees
        self.sample_with_replacement = sample_with_replacement
        self.attribute_with_replacement = attribute_with_replacement
        self.random_state = random_state
        self.trees = None   # Lista de árboles
        self.attribute_indices = None   # Lista de índices de atributos usados por cada árbol
        self.validator = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Entrena el ensemble utilizando los datos de entrada.
        """
        # Inicializa la lista de árboles y de índices de atributos
        self.trees = []
        self.attribute_indices = []
        
        # Convierte los DataFrames en arreglos de NumPy
        X = X.values
        y = y.values
        
        # Inicializa un generador de números aleatorios
        rng = np.random.default_rng(self.random_state)
        n_samples, n_attributes = X.shape
        
        for _ in range(self.n_trees):
            if self.sample_with_replacement:
                # Elige los índices de muestras con reemplazo
                sample_indices = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                # Crea un filtro para seleccionar muestras sin reemplazo
                sample_mask = rng.random(n_samples) > 0.5
                sample_indices = np.where(sample_mask)[0]
                
                if len(sample_indices) == 0:
                    # Si no se selecciona ninunga muestra, se eligen todas
                    sample_indices = rng.choice(n_samples, size=n_samples, replace=False)
            
            # Obtiene las muestras seleccionadas
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            if self.attribute_with_replacement:
                # Elige los índices de atributos con reemplazo
                attribute_indices = rng.choice(n_attributes, size=n_attributes, replace=True)
            else:
                # Crea un filtro para seleccionar atributos sin reemplazo
                attribute_mask = rng.random(n_attributes) > 0.5
                attribute_indices = np.where(attribute_mask)[0]
                
                if len(attribute_indices) == 0:
                    # Si no se selecciona ningún atributo, se eligen todos
                    attribute_indices = rng.choice(n_attributes, size=n_attributes, replace=False)
            
            # Selecciona los atributos elegidos para las muestras seleccionadas
            X_sample_attr = X_sample[:, attribute_indices]
            
            # Crea un árbol de decisión
            tree = DecisionTreeClassifier(random_state=self.random_state)
            tree.fit(X_sample_attr, y_sample)
            
            # Guarda el árbol y los índices de los atributos usados
            self.trees.append(tree)
            self.attribute_indices.append(attribute_indices)
            
    def classify(self, X: pd.DataFrame) -> np.ndarray:
        """
        Clasifica los datos de entrada utilizando el ensemble.
        """
        # Convierte DataFrame a arreglo de NumPy
        X = X.values
        classifications = []
        
        # Para cada árbol y sus atributos usados
        for tree, attr_indices in zip(self.trees, self.attribute_indices):
            # Selecciona los atributos usados por el árbol en las muestras
            X_tree = X[:, attr_indices]
            
            # Realiza la clasificación con el árbol
            tree_class = tree.predict(X_tree)
            classifications.append(tree_class)
        
        # Convierte la lista de clasificaciones a una matriz
        classifications = np.array(classifications)
        final_classifications = []
        
        # Para cada muestra, realiza la clasificación por mayoría
        for i in range(classifications.shape[1]):
            # Obtiene los votos de todos los árboles para la muestra i
            sample_votes = classifications[:, i]
            
            # Determina la clase mayoritaria
            unique_classes, counts = np.unique(sample_votes, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            final_classifications.append(majority_class)
        
        return np.array(final_classifications)
    
    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: int = 5
    ):
        """
        Realiza validación cruzada sobre los datos dados.
        """
        self.validator = Validator(folds=folds, random_state=self.random_state)
        
        # Realiza la validación cruzada
        results = self.validator.validate(self, X, y)
        return results
    
    def print_results(self):
        """
        Imprime los resulados de la validación.
        """
        self.validator.print_results()
    
    def save_validation_results(self):
        """
        Guarda los resultados de la validación en un archivo CSV.
        """
        results = self.validator.get_results()
        
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_data = {
            'Árboles': [self.n_trees],
            'Muestreo_con_reemplazo': [self.sample_with_replacement],
            'Atributos_con_reemplazo': [self.attribute_with_replacement],
            'Folds': [results['folds']],
            'Exactitud_promedio': [results['mean_accuracy']],
            'Error_promedio': [results['mean_error']],
            'Exactitud_desviación_estándar': [results['std_accuracy']],
            'Error_desviación_estándar': [results['std_error']],
            'Exactitud_mínima': [results['min_accuracy']],
            'Exactitud_máxima': [results['max_accuracy']]
        }
        
        # Guarda el resumen general de la validación
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{self.output_path}/validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, sep=',', header=True, index=False)
        print(f"Resumen guardado en: {summary_path}")
        
        # Guarda los detalles de cada fold
        details_df = pd.DataFrame(results['folds_details'])
        details_path = f"{self.output_path}/validation_details_{timestamp}.csv"
        details_df.to_csv(details_path, sep=',', header=True, index=False)
        print(f"Detalles de folds guardados en: {details_path}")