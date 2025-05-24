\
import streamlit as st
import pandas as pd
from config import TYPE_COLORS

def load_css():
    """Carga los estilos CSS"""
    st.markdown("""
    <style>
        .main-header {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #FF5B5B;
            text-align: center;
        }
        .sub-header {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #5B8CFF;
            margin-top: 0;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .pokemon-card {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .pokemon-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        .pokemon-img {
            width: 120px;
            height: 120px;
            margin: 0 auto;
            display: block;
        }
        .pokemon-name {
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .pokemon-type {
            margin-top: 5px;
            font-style: italic;
        }
        .stats-container {
            margin-top: 10px;
        }
        .stat-bar {
            height: 15px;
            margin-bottom: 5px;
            border-radius: 5px;
            position: relative;
        }
        .metric-card {
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        .type-badge {
            display: inline-block;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 12px;
            font-size: 0.8em;
            color: white;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            font-size: 0.85em;
            color: #777;
        }
    </style>
    """, unsafe_allow_html=True)

def show_pokemon_card(name, sprite_url, type1, type2, stats):
    """Muestra una tarjeta para un Pokémon"""
    type2_html = '</span>'
    if pd.notna(type2) and type2: # Verificar que type2 no sea NaN y no esté vacío
        type2_html = f'''<span class="type-badge" style="background-color: {TYPE_COLORS.get(type2, "#777")}">{type2}</span>'''

    st.markdown(f'''
    <div class="pokemon-card">
        <img src="{sprite_url}" class="pokemon-img" alt="{name}">
        <div class="pokemon-name">{name}</div>
        <div class="pokemon-type">
            <span class="type-badge" style="background-color: {TYPE_COLORS.get(type1, '#777')}">
                {type1}
            </span>
            {type2_html}
        </div>
        <div class="stats-container">
    ''', unsafe_allow_html=True)
    
    max_stat = max(stats.values()) if stats else 0
    for stat_name, stat_value in stats.items():
        percentage = (stat_value / max_stat) * 100 if max_stat > 0 else 0
        color = "#FF5B5B" if stat_name == "HP" else \
                "#F8D030" if stat_name in ["Attack", "Sp. Atk"] else \
                "#3D7DCA" if stat_name in ["Defense", "Sp. Def"] else \
                "#78C850"  # Speed
        
        st.markdown(f'''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 70px; text-align: right; padding-right: 10px; font-size: 0.85em;">{stat_name}</div>
            <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px;">
                <div style="width: {percentage}%; height: 15px; background-color: {color}; border-radius: 5px;"></div>
            </div>
            <div style="width: 40px; text-align: right; padding-left: 10px; font-size: 0.85em;">{stat_value}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def calculate_stat_advantage(a_stats, b_stats):
    """Calcula y muestra las ventajas estadísticas entre los Pokémon"""
    advantages = {}
    total_advantage = 0
    
    for stat, value_a in a_stats.items():
        value_b = b_stats[stat]
        diff = value_a - value_b
        advantages[stat] = diff
        total_advantage += diff
    
    return advantages, total_advantage
