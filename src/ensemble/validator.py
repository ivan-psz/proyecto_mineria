"""
Módulo que implementa la clase Validator para la validación estratificada con k folds.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Dict

class Validator:
    """
    Clase para realizar la validación estratificada cruzada con k folds sobre el ensemble.
    Atributos:
        folds (int): Número de folds para la validación cruzada.
        random_state (int): Semilla para aletoriedad.
        results (dict): Diccionario para almacenar los resultados de la validación.
    """
    def __init__(self, folds: int = 5, random_state: int = 42):
        """
        Inicializa el validaror con el número de folds y la semilla.
        """
        self.folds = folds
        self.random_state = random_state
        self.results = None
    
    def validate(
        self,
        ensemble,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Realiza la validación cruzada estatificada con k folds sobre el ensemble dado.
        """
        kfold = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        
        accuracies = []
        errors = []
        fold_details = []
        
        fold_num = 1
        for train_index, test_index in kfold.split(X, y):
            # Separa los datos en entrenamiento y prueba para este fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Entrena el ensemble con los datos de entrenamiento del fold
            ensemble.train(X_train, y_train)
            
            # Clasifica las clases para el conjunto de prueba
            y_class = ensemble.classify(X_test)
            
            # Calcula la exactitud y el error
            accuracy = float(np.mean(y_class == y_test.values)) * 100.00
            error = (100.00 - accuracy) / 100.00
            
            accuracies.append(accuracy)
            errors.append(error)
            
            # Guarda los detalles del fold
            fold_details.append({
                'fold': fold_num,
                'accuracy': accuracy,
                'error': error,
                'train_size': len(train_index),
                'test_size': len(test_index)
            })
            
            print(f"Fold {fold_num}/{self.folds} completado. Exactitud: {accuracy}%")
            fold_num += 1
            
        self.results = {
            'folds': self.folds,
            'mean_accuracy': np.mean(accuracies),
            'mean_error': np.mean(errors),
            'std_accuracy': np.std(accuracies),
            'std_error': np.std(errors),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'folds_details': fold_details
        }
    
    def print_results(self):
        """
        Muestra el resumen de la validación cruzada.
        """
        print("\n\t\t\tRESULTADOS DE VALIDACIÓN")
        print(f"Número de folds: {self.folds}")
        print(f"\nExactitud promedio: {self.results['mean_accuracy']}%")
        print(f"Error promedio: {self.results['mean_error']}")
        print(f"Exactitud mínima: {self.results['min_accuracy']}%")
        print(f"Exactitud máxima: {self.results['max_accuracy']}%")
        
        print("\nDETALLES POR FOLD:")
        for detail in self.results['folds_details']:
            print(f"Fold {detail['fold']}: Exactitud = {detail['accuracy']}%. Error = {detail['error']}")
            
    def get_results(self) -> Dict:
        """
        Devuelve el diccionario con los resultados de la validación.
        """
        return self.results