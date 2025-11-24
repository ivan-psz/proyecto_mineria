from pathlib import Path
from preprocessing import Preprocessor
from ensemble import EnsembleDelBienestar

# Rutas de entrada y salida
INPUT_PATH = './data/'
OUTPUT_PATH = './data/processed/'

def get_input_file():
    while True:
        file_path = input("\nIngrese el nombre del archivo de entrada: ").strip()
        path = Path(INPUT_PATH + file_path)
        
        if path.exists() and path.is_file():
            return str(path)
        else:
            print(f"Error: El archivo '{file_path}' no existe. Intente de nuevo.")

def change_file(preprocessor: Preprocessor):
    print("\n\t\t\tCAMBIAR EL ARCHIVO DE ENTRADA")
    print("Archivo actual:", preprocessor.input_path)
    
    new_file = get_input_file()
    
    try:
        preprocessor.set_input_path(new_file)
        print("\nArchivo cambiado exitosamente.")
        preprocessor.summarize_data()
    except Exception as e:
        print(f"\nError al cambiar el archivo: {e}")

def normalization_menu(preprocessor: Preprocessor):
    print("\n\t\t\tMÉTODOS DE NORMALIZACIÓN")
    print("1. Min-Max")
    print("2. Z-Score")
    print("3. Decimal Scaling")
    print("4. Todos")
    
    choice = input("\nSeleccione una opción: ").strip()
    
    if choice == '1':
        while True:
            lower = float(input("Ingrese el límite inferior: ").strip())
            upper = float(input("Ingrese el límite superior: ").strip())
            if lower < upper:
                normalized_df = preprocessor.normalize(method='min_max', lower_bound=lower, upper_bound=upper)
                print("Normalización Min-Max completada.")
                preprocessor.save_data(normalized_df, method='min_max')
                break
            else:
                print("Error: El límite inferior debe ser menor que el límite superior. Intente de nuevo.")
    elif choice == '2':
        normalized_df = preprocessor.normalize(method='z_score')
        print("Normalización Z-Score completada.")
        preprocessor.save_data(normalized_df, method='z_score')
    elif choice == '3':
        normalized_df = preprocessor.normalize(method='decimal_scaling')
        print("Normalización Decimal Scaling completada.")
        preprocessor.save_data(normalized_df, method='decimal_scaling')
    elif choice == '4':
        while True:
            lower = float(input("Ingrese el límite inferior para Min-Max: ").strip())
            upper = float(input("Ingrese el límite superior para Min-Max: ").strip())
            if lower < upper:
                normalized_df = preprocessor.normalize(method='min_max', lower_bound=lower, upper_bound=upper)
                print("Normalización Min-Max completada.")
                preprocessor.save_data(normalized_df, method='min_max')
                break
            else:
                print("Error: El límite inferior debe ser menor que el límite superior. Intente de nuevo.")
            
        normalized_df = preprocessor.normalize(method='z_score')
        print("Normalización Z-Score completada.")
        preprocessor.save_data(normalized_df, method='z_score')
                
        normalized_df = preprocessor.normalize(method='decimal_scaling')
        print("Normalización Decimal Scaling completada.")
        preprocessor.save_data(normalized_df, method='decimal_scaling')
    else:
        print("Opción inválida. Intente de nuevo.")
        
def discretization_menu(preprocessor: Preprocessor):
    print("\n\t\t\tDISCRETIZACIÓN BASADA EN ENTROPÍA")
    print("Discretizando usando cuartiles...\n")
    
    try:
        # Obtener las particiones discretizadas y los metadatos
        df_left, df_right, metadata = preprocessor.discretize()
        
        preprocessor.save_data(df_left, method='discretized_left')
        preprocessor.save_data(df_right, method='discretized_right')
        preprocessor.save_metadata(metadata)
        
        print("\nDiscretización completa:")
        print(f"\t- Mejor columna: {metadata['Índice_mejor_columna'][0]}")
        print(f"\t- Punto de corte: {metadata['Mejor_punto_corte'][0]}")
        print(f"\t- Ganancia de información: {metadata['Ganancia_información'][0]}")
        print(f"\t- Muestras en la partición izquierda: {len(df_left)}")
        print(f"\t- Muestras en la partición derecha: {len(df_right)}")
        
    except Exception as e:
        print(f"\nError durante la discretización: {e}")

def ensemble_menu(preprocessor: Preprocessor):
    print("\n\t\t\tENSEMBLE DEL BIENESTAR")
    print(f"Archivo de trabajo: {preprocessor.input_path}")
    
    try:
        n_trees = int(input("\nIngrese el número de árboles para el ensemble: ").strip())
        
        while True:
            print("\n¿Muestrear con reemplazo?")
            print("1. Sí")
            print("2. No")
            sample_choice = input("Seleccione una opción: ").strip()
            
            if sample_choice == '1':
                sample_with_replacement = True
                break
            elif sample_choice == '2':
                sample_with_replacement = False
                break
            else:
                print("Opción inválida. Intente de nuevo.")
        
        while True:
            print("\n¿Muestrear atributos con reemplazo?")
            print("1. Sí")
            print("2. No")
            attribute_choice = input("Seleccione una opción: ").strip()
            
            if attribute_choice == '1':
                attribute_with_replacement = True
                break
            elif attribute_choice == '2':
                attribute_with_replacement = False
                break
            else:
                print("Opción inválida. Intente de nuevo.")
        
        while True:
            folds = int(input("\nIngrese el número de folds para validación cruzada: ").strip())
            if folds > 3:
                break
            else:
                print("El número de folds debe ser mayor que 3. Intente de nuevo.")
        
        X = preprocessor.df.iloc[:, :-1]
        y = preprocessor.df.iloc[:, -1]
        
        print("\nCreando ensemble...")
        ensemble = EnsembleDelBienestar(
            output_path=OUTPUT_PATH,
            n_trees=n_trees,
            sample_with_replacement=sample_with_replacement,
            attribute_with_replacement=attribute_with_replacement,
            random_state=42
        )
        
        print(f"Validando con stratified {folds}-fold cross-validation...")
        ensemble.validate(X, y, folds=folds)
        ensemble.print_results()
        ensemble.save_validation_results()

    except ValueError as e:
        print(f"\nIngrese valores numéricos válidos. Error: {e}")
    except Exception as e:
        print(f"\nError durante la creación o validación del ensemble: {e}")

def main():
    print("\n\t\t\tPREPROCESAMIENTO DE DATOS")
    
    input_file = get_input_file()
    
    try:
        preprocessor = Preprocessor(input_path=input_file, output_path=OUTPUT_PATH)
        print("\nDatos cargados correctamente.")
        preprocessor.summarize_data()
    except Exception as e:
        print(f"\nError al cargar los datos: {e}")
        return
    
    while True:
        print("\n\t\t\tMENÚ PRINCIPAL")
        print("1. Normalización")
        print("2. Discretización")
        print("3. Ensemble")
        print("4. Cambiar archivo de trabajo")
        print("5. Salir")
        
        option = input("\nSeleccione una opción: ").strip()
        
        if option == '1':
            normalization_menu(preprocessor)
        elif option == '2':
            discretization_menu(preprocessor)
        elif option == '3':
            ensemble_menu(preprocessor)
        elif option == '4':
            change_file(preprocessor)
        elif option == '5':
            print("Finalizando ejecución.")
            break
        else:
            print("Opción inválida. Intente de nuevo.")
            
if __name__ == "__main__":
    main()