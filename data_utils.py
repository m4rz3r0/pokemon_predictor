import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from typing import Tuple, Dict, List, Union, Optional
from config import TYPE_COLORS

# Variable global para almacenar las columnas de características del modelo
MODEL_FEATURE_COLUMNS: List[str] = []

class PokemonBattleError(Exception):
    """Excepción base para errores del predictor de combates Pokémon"""
    pass

class DataValidationError(PokemonBattleError):
    """Excepción para errores de validación de datos"""
    pass

class ModelError(PokemonBattleError):
    """Excepción para errores del modelo"""
    pass

@st.cache_data
def load_pokemon_data() -> pd.DataFrame:
    """Carga los datos de Pokémon"""
    pokemon_df = pd.read_csv('data/all_pokemon_data.csv')
    pokemon_df['ID'] = pokemon_df['ID'].astype(str)
    validate_pokemon_data(pokemon_df)
    return pokemon_df

@st.cache_data
def load_combat_data() -> pd.DataFrame:
    """Carga los datos de combates"""
    combats_df = pd.read_csv('data/combats.csv')
    combats_df['First_pokemon'] = combats_df['First_pokemon'].astype(str)
    combats_df['Second_pokemon'] = combats_df['Second_pokemon'].astype(str)
    combats_df['Winner'] = combats_df['Winner'].astype(str)
    validate_combat_data(combats_df)
    return combats_df

@st.cache_data
def load_features_data() -> pd.DataFrame:
    """Carga los datos de características de combates"""
    features_df = pd.read_csv('data/combats_features.csv')
    id_columns = ['pokemon_A_id', 'pokemon_B_id', 'winner_id']
    for col in id_columns:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str)
    validate_features_data(features_df)
    return features_df

@st.cache_resource
def train_or_load_model(features_df: pd.DataFrame, force_retrain: bool = False) -> Pipeline:
    """Entrena un nuevo modelo o carga uno guardado"""
    model_path = 'pokemon_battle_model_xgboost.pkl'
    
    if os.path.exists(model_path) and not force_retrain:
        try:
            model = joblib.load(model_path)
            st.sidebar.success("Modelo cargado correctamente")
            return model
        except Exception as e:
            st.sidebar.error(f"Error al cargar el modelo: {e}")
            force_retrain = True
    
    if force_retrain or not os.path.exists(model_path):
        with st.spinner("Entrenando un nuevo modelo XGBoost..."):
            X = features_df.drop(['pokemon_A_id', 'pokemon_B_id', 'winner_id', 'A_Name', 'B_Name', 
                                  'A_SpriteURL', 'B_SpriteURL', 'target'], axis=1, errors='ignore')
            y = features_df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.sidebar.info(f"Modelo entrenado en {training_time:.2f} segundos")
            st.sidebar.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            joblib.dump(pipeline, model_path)
            return pipeline

def prepare_prediction_data(pokemon_a_id: str, pokemon_b_id: str, pokemon_df: pd.DataFrame, expected_feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Union[str, Optional[str]]]]:
    """Prepara los datos para la predicción"""
    pokemon_a = pokemon_df[pokemon_df['ID'] == pokemon_a_id].iloc[0]
    pokemon_b = pokemon_df[pokemon_df['ID'] == pokemon_b_id].iloc[0]
    
    features = {}
    stat_mapping = {
        'HP': 'hp',
        'Attack': 'attack',
        'Defense': 'defense',
        'Special Attack': 'special_attack',
        'Special Defense': 'special_defense',
        'Speed': 'speed'
    }
    
    for df_col, feat_col in stat_mapping.items():
        features[f'A_{feat_col}'] = pokemon_a[df_col]
        features[f'B_{feat_col}'] = pokemon_b[df_col]
        features[f'diff_{feat_col}'] = pokemon_a[df_col] - pokemon_b[df_col]
    
    for type_name in TYPE_COLORS.keys():
        features[f'A_type_1__{type_name}'] = 0
        features[f'A_type_2__{type_name}'] = 0
        features[f'B_type_1__{type_name}'] = 0
        features[f'B_type_2__{type_name}'] = 0
    
    features[f'A_type_1__{pokemon_a["Type1"]}'] = 1
    features[f'B_type_1__{pokemon_b["Type1"]}'] = 1
    
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
    
    prediction_features_df = pd.DataFrame([features])
    final_df = pd.DataFrame(0, index=[0], columns=expected_feature_columns, dtype=np.float64)

    for col in prediction_features_df.columns:
        if col in final_df.columns:
            final_df[col] = prediction_features_df[col].values
    
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
    
    return final_df, metadata

def get_historical_results(pokemon_a_id: str, pokemon_b_id: str, combats_df: pd.DataFrame) -> Tuple[int, int, int]:
    """Obtiene resultados históricos de combates entre los dos Pokémon"""
    result_a_first = combats_df[(combats_df['First_pokemon'] == pokemon_a_id) & 
                              (combats_df['Second_pokemon'] == pokemon_b_id)]
    result_b_first = combats_df[(combats_df['First_pokemon'] == pokemon_b_id) & 
                              (combats_df['Second_pokemon'] == pokemon_a_id)]
    
    a_wins = len(result_a_first[result_a_first['Winner'] == pokemon_a_id]) + \
             len(result_b_first[result_b_first['Winner'] == pokemon_a_id])
    b_wins = len(result_a_first[result_a_first['Winner'] == pokemon_b_id]) + \
             len(result_b_first[result_b_first['Winner'] == pokemon_b_id])
    total_matches = a_wins + b_wins
    
    return a_wins, b_wins, total_matches

def set_model_feature_columns(columns: List[str]):
    global MODEL_FEATURE_COLUMNS
    MODEL_FEATURE_COLUMNS = columns

def get_model_feature_columns() -> List[str]:
    return MODEL_FEATURE_COLUMNS

def validate_pokemon_data(df: pd.DataFrame) -> None:
    """Valida que el DataFrame de Pokémon tenga las columnas necesarias"""
    required_columns = ['ID', 'Name', 'HP', 'Attack', 'Defense', 'Special Attack', 
                       'Special Defense', 'Speed', 'Type1', 'SpriteURL']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataValidationError(f"Faltan columnas requeridas en datos de Pokémon: {missing_columns}")
    
    if df.empty:
        raise DataValidationError("El DataFrame de Pokémon está vacío")
    
    # Verificar que las estadísticas sean numéricas
    stat_columns = ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']
    for col in stat_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"La columna {col} debe ser numérica")

def validate_combat_data(df: pd.DataFrame) -> None:
    """Valida que el DataFrame de combates tenga las columnas necesarias"""
    required_columns = ['First_pokemon', 'Second_pokemon', 'Winner']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataValidationError(f"Faltan columnas requeridas en datos de combates: {missing_columns}")
    
    if df.empty:
        raise DataValidationError("El DataFrame de combates está vacío")

def validate_features_data(df: pd.DataFrame) -> None:
    """Valida que el DataFrame de características tenga las columnas necesarias"""
    required_columns = ['pokemon_A_id', 'pokemon_B_id', 'winner_id', 'target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataValidationError(f"Faltan columnas requeridas en datos de características: {missing_columns}")
    
    if df.empty:
        raise DataValidationError("El DataFrame de características está vacío")
