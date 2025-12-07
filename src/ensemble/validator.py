import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import Dict

class Validator:
    def __init__(self, folds: int = 5, random_state: int = 42):
        self.folds = folds
        self.random_state = random_state
        self.results = None
    
    def validate(self, ensemble, X: pd.DataFrame, y: pd.Series):
        kfold = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        
        accuracies = []
        errors = []
        fold_details = []
        
        fold_num = 1
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            ensemble.train(X_train, y_train)
            
            y_class = ensemble.classify(X_test)
            
            accuracy = float(np.mean(y_class == y_test.values)) * 100.00
            error = (100.00 - accuracy) / 100.00
            
            accuracies.append(accuracy)
            errors.append(error)
            
            fold_details.append({
                'Fold': fold_num,
                'Exactitud': accuracy,
                'Error': error,
                'Tamaño de entrenamiento': len(train_index),
                'Tamaño de prueba': len(test_index)
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
        print("\n\t\t\tRESULTADOS DE VALIDACIÓN")
        print(f"Número de folds: {self.folds}")
        print(f"\nExactitud promedio: {self.results['mean_accuracy']}%")
        print(f"Error promedio: {self.results['mean_error']}")
        print(f"Exactitud mínima: {self.results['min_accuracy']}%")
        print(f"Exactitud máxima: {self.results['max_accuracy']}%")
        
        print("\nDETALLES POR FOLD:")
        for detail in self.results['folds_details']:
            print(f"Fold {detail['Fold']}: Exactitud = {detail['Exactitud']}%. Error = {detail['Error']}")
            
    def get_results(self) -> Dict:
        return self.results