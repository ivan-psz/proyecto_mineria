import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .validator import Validator
from datetime import datetime

class EnsembleDelBienestar:
    def __init__(
        self,
        output_path: str,
        n_trees: int,
        sample_with_replacement: bool,
        attribute_with_replacement: bool,
        random_state: int = 42
    ):
        self.output_path = output_path
        self.n_trees = n_trees
        self.sample_with_replacement = sample_with_replacement
        self.attribute_with_replacement = attribute_with_replacement
        self.random_state = random_state
        self.trees = None   # Lista de árboles
        self.attribute_indices = None   # Lista de índices de atributos usados por cada árbol
        self.validator = None
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.trees = []
        self.attribute_indices = []
        
        # Convierte los DataFrames en arreglos de NumPy
        X = X.values
        y = y.values
        
        rng = np.random.default_rng(self.random_state)
        n_samples, n_attributes = X.shape
        
        for _ in range(self.n_trees):
            if self.sample_with_replacement:
                sample_indices = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                # Crea un filtro para seleccionar muestras sin reemplazo
                sample_filter = rng.random(n_samples) > 0.5
                sample_indices = np.where(sample_filter)[0]
                
                if len(sample_indices) == 0:
                    sample_indices = rng.choice(n_samples, size=n_samples, replace=False)
            
            # Obtiene las muestras seleccionadas
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            if self.attribute_with_replacement:
                attribute_indices = rng.choice(n_attributes, size=n_attributes, replace=True)
            else:
                # Crea un filtro para seleccionar atributos sin reemplazo
                attribute_mask = rng.random(n_attributes) > 0.5
                attribute_indices = np.where(attribute_mask)[0]
                
                if len(attribute_indices) == 0:
                    attribute_indices = rng.choice(n_attributes, size=n_attributes, replace=False)
            
            
            # Selecciona los atributos elegidos para las muestras seleccionadas
            X_sample_attr = X_sample[:, attribute_indices]
            
            tree = DecisionTreeClassifier(random_state=self.random_state)
            tree.fit(X_sample_attr, y_sample)
            
            self.trees.append(tree)
            self.attribute_indices.append(attribute_indices)
            
            """
            Líneas agregadas el jueves en la entrega:
            
            print("Árbol entrenado con:")
            print (attribute_indices)
            """
            
    def classify(self, X: pd.DataFrame) -> np.ndarray:
        # Convierte el DataFrame a arreglo de NumPy
        X = X.values
        classifications = []
        
        for tree, attr_indices in zip(self.trees, self.attribute_indices):
            # Selecciona los atributos usados por el árbol en las muestras
            X_tree = X[:, attr_indices]
            
            tree_class = tree.predict(X_tree)
            classifications.append(tree_class)
        
        classifications = np.array(classifications)
        final_classifications = []
        
        for i in range(classifications.shape[1]):
            sample_votes = classifications[:, i]
            
            # Determina la clase mayoritaria
            unique_classes, counts = np.unique(sample_votes, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            final_classifications.append(majority_class)
        
        return np.array(final_classifications)
    
    def validate(self, X: pd.DataFrame, y: pd.Series, folds: int = 5):
        self.validator = Validator(folds=folds, random_state=self.random_state)
        self.validator.validate(self, X, y)
    
    def print_results(self):
        self.validator.print_results()
    
    def save_validation_results(self):
        results = self.validator.get_results()
        
        if self.sample_with_replacement:
            sample_with_replacement_str = 'Sí'
        else:
            sample_with_replacement_str = 'No'
            
        if self.attribute_with_replacement:
            attribute_with_replacement_str = 'Sí'
        else:
            attribute_with_replacement_str = 'No'
        
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_data = {
            'Árboles': [self.n_trees],
            'Muestreo con reemplazo': [sample_with_replacement_str],
            'Atributos con reemplazo': [attribute_with_replacement_str],
            'Folds': [results['folds']],
            'Exactitud promedio': [results['mean_accuracy']],
            'Error promedio': [results['mean_error']],
            'Desviación estándar de exactitud': [results['std_accuracy']],
            'Desviación estándar de error': [results['std_error']],
            'Exactitud mínima': [results['min_accuracy']],
            'Exactitud máxima': [results['max_accuracy']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{self.output_path}/validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, sep=',', header=True, index=False)
        print(f"Resumen guardado en: {summary_path}")
        
        details_df = pd.DataFrame(results['folds_details'])
        details_path = f"{self.output_path}/validation_details_{timestamp}.csv"
        details_df.to_csv(details_path, sep=',', header=True, index=False)
        print(f"Detalles de folds guardados en: {details_path}")