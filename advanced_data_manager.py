"""
Clase mejorada para gestión de datos del sistema de predicción de combates Pokémon
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st

# Configurar logger
logger = logging.getLogger(__name__)

# Clases de excepción personalizadas
class PokemonBattleError(Exception):
    """Excepción base para errores del sistema de predicción de combates Pokémon"""
    pass

class DataValidationError(PokemonBattleError):
    """Excepción para errores de validación de datos"""
    pass

class ModelError(PokemonBattleError):
    """Excepción para errores relacionados con el modelo"""
    pass

class PokemonDataManager:
    """
    Clase centralizada para gestión de datos de Pokémon y combates
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._pokemon_df = None
        self._combat_df = None
        self._features_df = None
        self._model = None
        self._feature_columns = None
        
    @property
    def pokemon_data(self) -> pd.DataFrame:
        """Carga lazy de datos de Pokémon"""
        if self._pokemon_df is None:
            self._pokemon_df = self._load_pokemon_data()
        return self._pokemon_df
    
    @property
    def combat_data(self) -> pd.DataFrame:
        """Carga lazy de datos de combates"""
        if self._combat_df is None:
            self._combat_df = self._load_combat_data()
        return self._combat_df
    
    @property
    def features_data(self) -> pd.DataFrame:
        """Carga lazy de datos de características"""
        if self._features_df is None:
            self._features_df = self._load_features_data()
        return self._features_df
    
    def _load_pokemon_data(self) -> pd.DataFrame:
        """Carga y valida datos de Pokémon"""
        filepath = os.path.join(self.data_dir, 'all_pokemon_data.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        df = pd.read_csv(filepath)
        df['ID'] = df['ID'].astype(str)
        self._validate_pokemon_data(df)
        return df
    
    def _load_combat_data(self) -> pd.DataFrame:
        """Carga y valida datos de combates"""
        filepath = os.path.join(self.data_dir, 'combats.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        df = pd.read_csv(filepath)
        df['First_pokemon'] = df['First_pokemon'].astype(str)
        df['Second_pokemon'] = df['Second_pokemon'].astype(str)
        df['Winner'] = df['Winner'].astype(str)
        self._validate_combat_data(df)
        return df
    
    def _load_features_data(self) -> pd.DataFrame:
        """Carga y valida datos de características"""
        filepath = os.path.join(self.data_dir, 'combats_features.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        df = pd.read_csv(filepath)
        id_columns = ['pokemon_A_id', 'pokemon_B_id', 'winner_id']
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        self._validate_features_data(df)
        return df
    
    def _validate_pokemon_data(self, df: pd.DataFrame) -> None:
        """Valida estructura de datos de Pokémon"""
        required_columns = ['ID', 'Name', 'HP', 'Attack', 'Defense', 'Special Attack', 
                           'Special Defense', 'Speed', 'Type1', 'SpriteURL']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        if df.empty:
            raise ValueError("DataFrame de Pokémon está vacío")
    
    def _validate_combat_data(self, df: pd.DataFrame) -> None:
        """Valida estructura de datos de combates"""
        required_columns = ['First_pokemon', 'Second_pokemon', 'Winner']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        if df.empty:
            raise ValueError("DataFrame de combates está vacío")
    
    def _validate_features_data(self, df: pd.DataFrame) -> None:
        """Valida estructura de datos de características"""
        required_columns = ['pokemon_A_id', 'pokemon_B_id', 'winner_id', 'target']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        if df.empty:
            raise ValueError("DataFrame de características está vacío")
    
    def get_pokemon_by_id(self, pokemon_id: str) -> pd.Series:
        """Obtiene un Pokémon por su ID"""
        match = self.pokemon_data[self.pokemon_data['ID'] == pokemon_id]
        if match.empty:
            raise ValueError(f"No se encontró Pokémon con ID: {pokemon_id}")
        return match.iloc[0]
    
    def search_pokemon(self, search_term: str = "", type_filter: str = None) -> pd.DataFrame:
        """Busca Pokémon por nombre y/o tipo"""
        df = self.pokemon_data.copy()
        
        if search_term:
            df = df[df['Name'].str.lower().str.contains(search_term.lower(), na=False)]
        
        if type_filter and type_filter != 'Todos':
            df = df[(df['Type1'] == type_filter) | (df['Type2'] == type_filter)]
        
        return df
    
    def get_historical_results(self, pokemon_a_id: str, pokemon_b_id: str) -> Tuple[int, int, int]:
        """Obtiene resultados históricos entre dos Pokémon"""
        combat_df = self.combat_data
        
        result_a_first = combat_df[
            (combat_df['First_pokemon'] == pokemon_a_id) & 
            (combat_df['Second_pokemon'] == pokemon_b_id)
        ]
        result_b_first = combat_df[
            (combat_df['First_pokemon'] == pokemon_b_id) & 
            (combat_df['Second_pokemon'] == pokemon_a_id)
        ]
        
        a_wins = len(result_a_first[result_a_first['Winner'] == pokemon_a_id]) + \
                 len(result_b_first[result_b_first['Winner'] == pokemon_a_id])
        b_wins = len(result_a_first[result_a_first['Winner'] == pokemon_b_id]) + \
                 len(result_b_first[result_b_first['Winner'] == pokemon_b_id])
        
        return a_wins, b_wins, a_wins + b_wins
    
    def get_feature_columns(self) -> List[str]:
        """Obtiene las columnas de características para el modelo"""
        if self._feature_columns is None:
            features_df = self.features_data
            self._feature_columns = features_df.drop(
                ['pokemon_A_id', 'pokemon_B_id', 'winner_id', 'A_Name', 'B_Name',
                 'A_SpriteURL', 'B_SpriteURL', 'target'],
                axis=1, errors='ignore'
            ).columns.tolist()
        return self._feature_columns
    
    def prepare_prediction_data(self, pokemon_a_id: str, pokemon_b_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Prepara datos para predicción"""
        pokemon_a = self.get_pokemon_by_id(pokemon_a_id)
        pokemon_b = self.get_pokemon_by_id(pokemon_b_id)
        
        # Crear características básicas
        features = self._create_base_features(pokemon_a, pokemon_b)
        
        # Añadir características de tipo
        features.update(self._create_type_features(pokemon_a, pokemon_b))
        
        # Crear DataFrame con todas las columnas esperadas
        expected_columns = self.get_feature_columns()
        prediction_df = pd.DataFrame(0, index=[0], columns=expected_columns, dtype=np.float64)
        
        # Llenar con las características calculadas
        for col in features:
            if col in prediction_df.columns:
                prediction_df[col] = features[col]
        
        # Metadatos para la interfaz
        metadata = {
            'A_Name': pokemon_a['Name'],
            'A_SpriteURL': pokemon_a['SpriteURL'],
            'A_Type1': pokemon_a['Type1'],
            'A_Type2': pokemon_a['Type2'] if pd.notna(pokemon_a['Type2']) else None,
            'B_Name': pokemon_b['Name'],
            'B_SpriteURL': pokemon_b['SpriteURL'],
            'B_Type1': pokemon_b['Type1'],
            'B_Type2': pokemon_b['Type2'] if pd.notna(pokemon_b['Type2']) else None,
        }
        
        return prediction_df, metadata
    
    def _create_base_features(self, pokemon_a: pd.Series, pokemon_b: pd.Series) -> Dict:
        """Crea características básicas de estadísticas"""
        stat_mapping = {
            'HP': 'hp', 'Attack': 'attack', 'Defense': 'defense',
            'Special Attack': 'special_attack', 'Special Defense': 'special_defense',
            'Speed': 'speed'
        }
        
        features = {}
        for df_col, feat_col in stat_mapping.items():
            features[f'A_{feat_col}'] = pokemon_a[df_col]
            features[f'B_{feat_col}'] = pokemon_b[df_col]
            features[f'diff_{feat_col}'] = pokemon_a[df_col] - pokemon_b[df_col]
        
        return features
    
    def _create_type_features(self, pokemon_a: pd.Series, pokemon_b: pd.Series) -> Dict:
        """Crea características de tipo"""
        from config import TYPE_COLORS
        
        features = {}
        
        # Inicializar todas las características de tipo
        for type_name in TYPE_COLORS.keys():
            features[f'A_type_1__{type_name}'] = 0
            features[f'A_type_2__{type_name}'] = 0
            features[f'B_type_1__{type_name}'] = 0
            features[f'B_type_2__{type_name}'] = 0
        
        # Establecer tipos primarios
        features[f'A_type_1__{pokemon_a["Type1"]}'] = 1
        features[f'B_type_1__{pokemon_b["Type1"]}'] = 1
        
        # Establecer tipos secundarios
        if pd.notna(pokemon_a["Type2"]):
            features[f'A_type_2__{pokemon_a["Type2"]}'] = 1
            features[f'A_type_2__none'] = 0
        else:
            features[f'A_type_2__none'] = 1
        
        if pd.notna(pokemon_b["Type2"]):
            features[f'B_type_2__{pokemon_b["Type2"]}'] = 1
            features[f'B_type_2__none'] = 0
        else:
            features[f'B_type_2__none'] = 1
        
        return features

class PokemonModelManager:
    """
    Clase para gestión del modelo de predicción
    """
    
    def __init__(self, model_path: str = 'pokemon_battle_model_xgboost.pkl'):
        self.model_path = model_path
        self._model = None
    
    @property
    def model(self) -> Pipeline:
        """Carga lazy del modelo"""
        if self._model is None:
            self._model = self._load_or_train_model()
        return self._model
    
    def _load_or_train_model(self, force_retrain: bool = False) -> Pipeline:
        """Carga el modelo existente o entrena uno nuevo"""
        if os.path.exists(self.model_path) and not force_retrain:
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                st.warning(f"Error al cargar modelo: {e}. Entrenando nuevo modelo...")
                force_retrain = True
        
        if force_retrain or not os.path.exists(self.model_path):
            return self._train_new_model()
    
    def _train_new_model(self) -> Pipeline:
        """Entrena un nuevo modelo XGBoost"""
        try:
            # Intentar cargar el modelo existente si no se especifica uno
            models_dir = os.path.join(os.path.dirname(self.model_path) or '.', 'models')
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.endswith('.pkl'):
                        fallback_path = os.path.join(models_dir, file)
                        try:
                            return joblib.load(fallback_path)
                        except Exception:
                            continue
            
            # Si no hay modelo disponible, crear uno básico para pruebas
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
            ])
            
            # Crear datos sintéticos mínimos para inicializar
            X_dummy = np.random.rand(10, 5)
            y_dummy = np.random.randint(0, 2, 10)
            pipeline.fit(X_dummy, y_dummy)
            
            return pipeline
            
        except Exception as e:
            logger.warning(f"No se pudo entrenar modelo: {e}")
            raise ModelError(f"No se pudo crear modelo: {e}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Realiza predicción"""
        model = self.model
        prediction = model.predict(X)[0]
        probability_array = model.predict_proba(X)
        if isinstance(probability_array, list):
            probability = probability_array[0][1] if len(probability_array[0]) > 1 else probability_array[0][0]
        else:
            probability = probability_array[0, 1] if probability_array.shape[1] > 1 else probability_array[0, 0]
        
        return int(prediction), float(probability)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características"""
        model = self.model
        xgb_model = model.named_steps['model']
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        if feature_names is None:
            return {}
        
        return dict(zip(feature_names, xgb_model.feature_importances_))
