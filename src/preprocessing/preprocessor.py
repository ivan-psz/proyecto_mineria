import pandas as pd
from datetime import datetime
from .normalizer import Normalizer
from .discretizer import Discretizer
from typing import Tuple, Dict

class Preprocessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        
        self.df = pd.read_csv(filepath_or_buffer=self.input_path, sep=',', header=None, index_col=False)
        
        self.normalizer = Normalizer()
        self.discretizer = Discretizer(self.df)
    
    def set_input_path(self, new_path: str):
        self.input_path = new_path
        self.df = pd.read_csv(filepath_or_buffer=self.input_path, sep=',', header=None, index_col=False)
        
        self.discretizer.set_input_path(new_path)
        
    def summarize_data(self):
        print("\nResumen de los datos:")
        print(f"\tNúmero de filas: {self.df.shape[0]}")
        print(f"\tNúmero de columnas: {self.df.shape[1]}")
        
    def normalize(self, method: str, **kwargs) -> pd.DataFrame:
        if method == 'min_max':
            lower_bound = kwargs.get('lower_bound', 0.0)
            upper_bound = kwargs.get('upper_bound', 10.0)
            return self.normalizer.min_max(self.df, lower_bound, upper_bound)
        elif method == 'z_score':
            return self.normalizer.z_score(self.df)
        else:
            return self.normalizer.decimal_scaling(self.df)
        
    def discretize(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        # Calcula la mejor división
        self.discretizer.compute_info_gain()
        # Particiona y obtiene los metadatos
        df_left, df_right, metadata = self.discretizer.discretize()
        
        return df_left, df_right, metadata
        
    def save_data(self, df: pd.DataFrame, method: str):
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.output_path}/{method}_{timestamp}.csv"
        df.to_csv(file_path, sep=',', header=False, index=False)
        
        print(f"Datos guardados en: {self.output_path}")
        
    def save_metadata(self, metadata: Dict):
        # Genera un timestamp para el nombre del archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.output_path}/metadata_{timestamp}.csv"
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(file_path, sep=',', header=True, index=False)
        
        print(f"Metadatos guardados en: {self.output_path}")