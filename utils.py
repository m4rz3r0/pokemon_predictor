# utils.py - Funciones auxiliares mejoradas
import pandas as pd
from typing import Dict, List, Tuple, Optional

def create_pokemon_selector(pokemon_df: pd.DataFrame, key_suffix: str, label: str) -> str:
    """
    Función mejorada para seleccionar Pokémon con búsqueda y filtros
    """
    import streamlit as st
    
    # Búsqueda por nombre
    search_term = st.text_input(f"Buscar {label}", key=f"search_{key_suffix}")
    
    # Filtro por tipo
    all_types = sorted(pokemon_df['Type1'].dropna().unique())
    selected_type = st.selectbox(
        f"Filtrar por tipo ({label})", 
        options=['Todos'] + all_types, 
        key=f"type_filter_{key_suffix}"
    )
    
    # Aplicar filtros
    filtered_df = pokemon_df.copy()
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Name'].str.lower().str.contains(search_term.lower(), na=False)
        ]
    
    if selected_type != 'Todos':
        filtered_df = filtered_df[
            (filtered_df['Type1'] == selected_type) | 
            (filtered_df['Type2'] == selected_type)
        ]
    
    if filtered_df.empty:
        st.warning(f"No se encontraron Pokémon que coincidan con los filtros para {label}")
        return None
    
    # Selector principal
    pokemon_id = st.selectbox(
        f'Selecciona {label}',
        options=filtered_df['ID'].tolist(),
        format_func=lambda x: get_pokemon_name(x, filtered_df),
        key=f'pokemon_{key_suffix}'
    )
    
    return pokemon_id

def get_pokemon_name(pokemon_id: str, pokemon_df: pd.DataFrame) -> str:
    """Obtiene el nombre de un Pokémon por su ID"""
    match = pokemon_df[pokemon_df['ID'] == pokemon_id]
    return match['Name'].iloc[0] if not match.empty else str(pokemon_id)

def format_stat_comparison(stats_a: Dict, stats_b: Dict, name_a: str, name_b: str) -> pd.DataFrame:
    """
    Crea un DataFrame comparativo de estadísticas con formato mejorado
    """
    comparison_data = []
    
    for stat in stats_a.keys():
        val_a = stats_a[stat]
        val_b = stats_b[stat]
        diff = val_a - val_b
        
        comparison_data.append({
            'Estadística': stat,
            name_a: val_a,
            name_b: val_b,
            'Diferencia': diff,
            'Ventaja': name_a if diff > 0 else name_b if diff < 0 else 'Empate'
        })
    
    return pd.DataFrame(comparison_data)

def generate_battle_insights(
    pokemon_a: pd.Series, 
    pokemon_b: pd.Series, 
    type_effectiveness: Tuple[float, float],
    historical_data: Optional[Tuple[int, int, int]] = None
) -> List[str]:
    """
    Genera insights inteligentes sobre el combate
    """
    insights = []
    
    # Calcular estadísticas totales
    stats_a = {
        'HP': pokemon_a['HP'], 'Attack': pokemon_a['Attack'], 
        'Defense': pokemon_a['Defense'], 'Special Attack': pokemon_a['Special Attack'],
        'Special Defense': pokemon_a['Special Defense'], 'Speed': pokemon_a['Speed']
    }
    stats_b = {
        'HP': pokemon_b['HP'], 'Attack': pokemon_b['Attack'], 
        'Defense': pokemon_b['Defense'], 'Special Attack': pokemon_b['Special Attack'],
        'Special Defense': pokemon_b['Special Defense'], 'Speed': pokemon_b['Speed']
    }
    
    total_stats_a = sum(stats_a.values())
    total_stats_b = sum(stats_b.values())
    
    # Insight de estadísticas base
    if abs(total_stats_a - total_stats_b) > 50:
        stronger = pokemon_a['Name'] if total_stats_a > total_stats_b else pokemon_b['Name']
        stronger_total = max(total_stats_a, total_stats_b)
        weaker_total = min(total_stats_a, total_stats_b)
        insights.append(
            f"🔢 **{stronger}** tiene estadísticas base superiores "
            f"({stronger_total} vs {weaker_total} puntos totales)"
        )
    
    # Insight de velocidad
    speed_diff = stats_a['Speed'] - stats_b['Speed']
    if abs(speed_diff) > 20:
        faster = pokemon_a['Name'] if speed_diff > 0 else pokemon_b['Name']
        insights.append(f"⚡ **{faster}** tiene ventaja significativa en velocidad")
    
    # Insight de efectividad de tipo
    a_vs_b_mult, b_vs_a_mult = type_effectiveness
    if a_vs_b_mult > 1.5:
        insights.append(f"🎯 **{pokemon_a['Name']}** tiene ventaja de tipo significativa")
    elif b_vs_a_mult > 1.5:
        insights.append(f"🎯 **{pokemon_b['Name']}** tiene ventaja de tipo significativa")
    
    # Insight histórico
    if historical_data and historical_data[2] > 5:  # total_matches > 5
        a_wins, b_wins, total = historical_data
        if a_wins > b_wins * 1.5:
            insights.append(f"📊 **{pokemon_a['Name']}** domina históricamente ({a_wins}/{total} victorias)")
        elif b_wins > a_wins * 1.5:
            insights.append(f"📊 **{pokemon_b['Name']}** domina históricamente ({b_wins}/{total} victorias)")
    
    # Insight de roles de combate
    if stats_a['Attack'] > stats_a['Special Attack'] + 20:
        role_a = "atacante físico"
    elif stats_a['Special Attack'] > stats_a['Attack'] + 20:
        role_a = "atacante especial"
    else:
        role_a = "atacante mixto"
    
    if stats_b['Attack'] > stats_b['Special Attack'] + 20:
        role_b = "atacante físico"
    elif stats_b['Special Attack'] > stats_b['Attack'] + 20:
        role_b = "atacante especial"
    else:
        role_b = "atacante mixto"
    
    insights.append(f"⚔️ **{pokemon_a['Name']}** es {role_a}, **{pokemon_b['Name']}** es {role_b}")
    
    return insights

def create_confidence_display(confidence: float, winner_name: str) -> str:
    """Crea un HTML estilizado para mostrar la confianza de la predicción"""
    if confidence > 0.75:
        color = "#198754"
        bg_color = "rgba(25, 135, 84, 0.15)"
        confidence_text = "con alta confianza"
        icon = "🏆"
    elif confidence > 0.6:
        color = "#0d6efd"
        bg_color = "rgba(13, 110, 253, 0.15)"
        confidence_text = "probablemente"
        icon = "🎯"
    else:
        color = "#ffc107"
        bg_color = "rgba(255, 193, 7, 0.15)"
        confidence_text = "por un pequeño margen"
        icon = "⚖️"
    
    return f"""
    <div style="
        background-color: {bg_color}; 
        border-left: 4px solid {color}; 
        color: {color}; 
        padding: 20px; 
        border-radius: 10px; 
        margin: 20px 0;
    ">
        <h2 style="margin: 0; color: {color};">
            {icon} Predicción: <b>{winner_name}</b> ganará {confidence_text} ({confidence:.1%})
        </h2>
    </div>
    """
