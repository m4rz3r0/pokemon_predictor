\
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from config import TYPE_COLORS # Assuming TYPE_COLORS is in config.py
from plot_manager import plot_manager

def save_plot(fig, filename, subfolder="", metadata=None):
    """Guarda automáticamente los plots en la carpeta plots/ usando el plot manager"""
    if not subfolder:
        subfolder = "custom"
    
    return plot_manager.save_plot(fig, filename, subfolder, metadata)

def plot_match_probability(proba, name_a, name_b, save_plot_flag=True):
    """Genera un gráfico de probabilidad para el combate"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Lógica corregida: Azul para el ganador más probable, Rojo para el menos probable
    # proba es la probabilidad de que gane name_a
    
    if proba >= 0.5:
        # name_a es más probable de ganar (azul a la izquierda)
        ax.barh(0, proba, color='#5B8CFF', height=0.5)  # Probabilidad de A (azul)
        ax.barh(0, 1-proba, left=proba, color='#FF5B5B', height=0.5)  # Probabilidad de B (rojo)
        # Nombres: A va a la izquierda (azul), B a la derecha (rojo)
        ax.text(0.02, 0, name_a, ha='left', va='center', color='white', fontweight='bold')
        ax.text(0.98, 0, name_b, ha='right', va='center', color='white', fontweight='bold')
    else:
        # name_b es más probable de ganar (invertir colores)
        ax.barh(0, 1-proba, color='#5B8CFF', height=0.5)  # Probabilidad de B (azul)
        ax.barh(0, proba, left=1-proba, color='#FF5B5B', height=0.5)  # Probabilidad de A (rojo)
        # Nombres: B va a la izquierda (azul), A a la derecha (rojo)
        ax.text(0.02, 0, name_b, ha='left', va='center', color='white', fontweight='bold')
        ax.text(0.98, 0, name_a, ha='right', va='center', color='white', fontweight='bold')
    
    
    # Mostrar porcentaje del ganador más probable
    if proba >= 0.5:
        ax.text(proba/2, 0, f"{proba:.1%}", ha='center', va='center', color='white', fontweight='bold')
    else:
        ax.text((1-proba)/2, 0, f"{1-proba:.1%}", ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_title("Probabilidad de Victoria", pad=10)
    
    # Guardar automáticamente el plot
    if save_plot_flag:
        metadata = {
            "pokemon_a": name_a,
            "pokemon_b": name_b,
            "probability_a": proba,
            "probability_b": 1-proba
        }
        save_plot(fig, f"match_probability_{name_a}_vs_{name_b}.png", "match_predictions", metadata)
    
    return fig

def plot_feature_importance(model, feature_names, top_n=10, save_plot_flag=True):
    """Genera un gráfico de importancia de características"""
    xgb_model = model.named_steps['model']
    importances = xgb_model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
    ax.set_title(f'Las {top_n} características más importantes')
    ax.set_xlabel('Importancia relativa')
    ax.set_ylabel('')
    
    # Guardar automáticamente el plot
    if save_plot_flag:
        metadata = {
            "top_n": top_n,
            "total_features": len(feature_names),
            "top_features": feat_imp['Feature'].tolist()
        }
        save_plot(fig, f"feature_importance_top_{top_n}.png", "model_analysis", metadata)
    
    return fig

def generate_type_effectiveness_chart(type1_a, type2_a, type1_b, type2_b):
    """Genera una tabla de efectividad de tipos para los Pokémon seleccionados"""
    type_chart = {
        'Normal': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                  'Fighting': 1, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                  'Bug': 1, 'Rock': 0.5, 'Ghost': 0, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5, 'Fairy': 1},
        'Fire': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1, 'Grass': 2, 'Ice': 2, 
                'Fighting': 1, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                'Bug': 2, 'Rock': 0.5, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 2, 'Fairy': 1},
        'Water': {'Normal': 1, 'Fire': 2, 'Water': 0.5, 'Electric': 1, 'Grass': 0.5, 'Ice': 1, 
                 'Fighting': 1, 'Poison': 1, 'Ground': 2, 'Flying': 1, 'Psychic': 1, 
                 'Bug': 1, 'Rock': 2, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 1, 'Fairy': 1},
        'Electric': {'Normal': 1, 'Fire': 1, 'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ice': 1, 
                    'Fighting': 1, 'Poison': 1, 'Ground': 0, 'Flying': 2, 'Psychic': 1, 
                    'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 1, 'Fairy': 1},
        'Grass': {'Normal': 1, 'Fire': 0.5, 'Water': 2, 'Electric': 1, 'Grass': 0.5, 'Ice': 1, 
                 'Fighting': 1, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Psychic': 1, 
                 'Bug': 0.5, 'Rock': 2, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 0.5, 'Fairy': 1},
        'Ice': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1, 'Grass': 2, 'Ice': 0.5, 
               'Fighting': 1, 'Poison': 1, 'Ground': 2, 'Flying': 2, 'Psychic': 1, 
               'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 1, 'Steel': 0.5, 'Fairy': 1},
        'Fighting': {'Normal': 2, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 2, 
                    'Fighting': 1, 'Poison': 0.5, 'Ground': 1, 'Flying': 0.5, 'Psychic': 0.5, 
                    'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dragon': 1, 'Dark': 2, 'Steel': 2, 'Fairy': 0.5},
        'Poison': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 2, 'Ice': 1, 
                  'Fighting': 1, 'Poison': 0.5, 'Ground': 0.5, 'Flying': 1, 'Psychic': 1, 
                  'Bug': 1, 'Rock': 0.5, 'Ghost': 0.5, 'Dragon': 1, 'Dark': 1, 'Steel': 0, 'Fairy': 2},
        'Ground': {'Normal': 1, 'Fire': 2, 'Water': 1, 'Electric': 2, 'Grass': 0.5, 'Ice': 1, 
                  'Fighting': 1, 'Poison': 2, 'Ground': 1, 'Flying': 0, 'Psychic': 1, 
                  'Bug': 0.5, 'Rock': 2, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 2, 'Fairy': 1},
        'Flying': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 0.5, 'Grass': 2, 'Ice': 1, 
                  'Fighting': 2, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                  'Bug': 2, 'Rock': 0.5, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5, 'Fairy': 1},
        'Psychic': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                   'Fighting': 2, 'Poison': 2, 'Ground': 1, 'Flying': 1, 'Psychic': 0.5, 
                   'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 1, 'Dark': 0, 'Steel': 0.5, 'Fairy': 1},
        'Bug': {'Normal': 1, 'Fire': 0.5, 'Water': 1, 'Electric': 1, 'Grass': 2, 'Ice': 1, 
               'Fighting': 0.5, 'Poison': 0.5, 'Ground': 1, 'Flying': 0.5, 'Psychic': 2, 
               'Bug': 1, 'Rock': 1, 'Ghost': 0.5, 'Dragon': 1, 'Dark': 2, 'Steel': 0.5, 'Fairy': 0.5},
        'Rock': {'Normal': 1, 'Fire': 2, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 2, 
                'Fighting': 0.5, 'Poison': 1, 'Ground': 0.5, 'Flying': 2, 'Psychic': 1, 
                'Bug': 2, 'Rock': 1, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5, 'Fairy': 1},
        'Ghost': {'Normal': 0, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                 'Fighting': 1, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 2, 
                 'Bug': 1, 'Rock': 1, 'Ghost': 2, 'Dragon': 1, 'Dark': 0.5, 'Steel': 1, 'Fairy': 1},
        'Dragon': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                  'Fighting': 1, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                  'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 1, 'Steel': 0.5, 'Fairy': 0},
        'Dark': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                'Fighting': 0.5, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 2, 
                'Bug': 1, 'Rock': 1, 'Ghost': 2, 'Dragon': 1, 'Dark': 0.5, 'Steel': 1, 'Fairy': 0.5},
        'Steel': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Grass': 1, 'Ice': 2, 
                 'Fighting': 1, 'Poison': 1, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                 'Bug': 1, 'Rock': 2, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5, 'Fairy': 2},
        'Fairy': {'Normal': 1, 'Fire': 0.5, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 
                 'Fighting': 2, 'Poison': 0.5, 'Ground': 1, 'Flying': 1, 'Psychic': 1, 
                 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5, 'Fairy': 1}
    }
    
    types_a = [type1_a]
    if type2_a:
        types_a.append(type2_a)
    
    types_b = [type1_b]
    if type2_b:
        types_b.append(type2_b)
    
    a_vs_b_multiplier = 1.0
    for attack_type in types_a:
        for defense_type in types_b:
            a_vs_b_multiplier *= type_chart.get(attack_type, {}).get(defense_type, 1.0)
    
    b_vs_a_multiplier = 1.0
    for attack_type in types_b:
        for defense_type in types_a:
            b_vs_a_multiplier *= type_chart.get(attack_type, {}).get(defense_type, 1.0)
    
    a_vs_b_text = "Normal"
    if a_vs_b_multiplier > 1:
        a_vs_b_text = f"Súper Efectivo (x{a_vs_b_multiplier})"
    elif a_vs_b_multiplier < 1 and a_vs_b_multiplier > 0:
        a_vs_b_text = f"No muy Efectivo (x{a_vs_b_multiplier})"
    elif a_vs_b_multiplier == 0:
        a_vs_b_text = "Sin efecto"
    
    b_vs_a_text = "Normal"
    if b_vs_a_multiplier > 1:
        b_vs_a_text = f"Súper Efectivo (x{b_vs_a_multiplier})"
    elif b_vs_a_multiplier < 1 and b_vs_a_multiplier > 0:
        b_vs_a_text = f"No muy Efectivo (x{b_vs_a_multiplier})"
    elif b_vs_a_multiplier == 0:
        b_vs_a_text = "Sin efecto"
    
    return a_vs_b_text, b_vs_a_text, a_vs_b_multiplier, b_vs_a_multiplier
