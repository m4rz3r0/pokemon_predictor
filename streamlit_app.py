import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from config import PAGE_CONFIG, TYPE_COLORS
from advanced_data_manager import PokemonDataManager, PokemonModelManager
from data_utils import (
    set_model_feature_columns, get_model_feature_columns
)
from ui_utils import load_css, show_pokemon_card, calculate_stat_advantage
from plot_utils import plot_match_probability, plot_feature_importance, generate_type_effectiveness_chart
from shap_utils import explain_prediction_shap
from utils import format_stat_comparison, generate_battle_insights
from plot_manager import plot_manager

# Configuración de la página
st.set_page_config(**PAGE_CONFIG)

def save_streamlit_plot(fig, filename, subfolder="streamlit_plots", metadata=None):
    """Guarda automáticamente los plots generados en streamlit usando el plot manager"""
    return plot_manager.save_plot(fig, filename, subfolder, metadata)

# --- Inicializar gestores de datos ---
@st.cache_resource
def get_data_manager():
    """Inicializa y cachea el gestor de datos"""
    return PokemonDataManager()

@st.cache_resource  
def get_model_manager():
    """Inicializa y cachea el gestor de modelo"""
    return PokemonModelManager()

# --- Interfaz Streamlit ---
def main():
    # Cargar estilos CSS
    load_css()
    
    # Título principal
    st.markdown("<h1 class='main-header'>⚔️ Predictor de Combates Pokémon</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Sistema de predicción avanzado basado en datos reales de combates</p>", unsafe_allow_html=True)
    
    # Configuración de la barra lateral
    st.sidebar.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/master-ball.png", width=50)
    st.sidebar.title("Configuración")
    
    # Cargar datos usando gestores avanzados
    try:
        data_manager = get_data_manager()
        model_manager = get_model_manager()
        
        # Acceso a datos a través del data manager
        pokemon_df = data_manager.pokemon_data
        combats_df = data_manager.combat_data
        features_df = data_manager.features_data

        # Configurar columnas del modelo
        model_feature_columns = data_manager.get_feature_columns()
        set_model_feature_columns(model_feature_columns)
        
        # Acceso al modelo a través del gestor
        force_retrain = st.sidebar.checkbox("Forzar reentrenamiento del modelo", False)
        # El modelo se carga automáticamente mediante el model_manager cuando se necesita
        
        # Tabs para organizar la interfaz
        tab1, tab2, tab3 = st.tabs(["Predicción de Combate", "Análisis del Modelo", "Acerca de"])
        
        with tab1:
            st.markdown("<h2 class='sub-header'>Selecciona dos Pokémon para predecir el resultado</h2>", unsafe_allow_html=True)
            
            # Selección de Pokémon
            col1, col2 = st.columns(2)
            
            with col1:
                # Opciones para filtrar
                search_term_a = st.text_input("Buscar Pokémon A", key="search_a")
                filtered_pokemon_a = pokemon_df
                if search_term_a:
                    filtered_pokemon_a = pokemon_df[pokemon_df['Name'].str.lower().str.contains(search_term_a.lower())]
                
                pokemon_a_id = st.selectbox(
                    'Selecciona el primer Pokémon',
                    options=filtered_pokemon_a['ID'].tolist(),
                    format_func=lambda x: filtered_pokemon_a[filtered_pokemon_a['ID'] == x]['Name'].iloc[0] if x in filtered_pokemon_a['ID'].values else x,
                    key='p1'
                )
            
            with col2:
                search_term_b = st.text_input("Buscar Pokémon B", key="search_b")
                filtered_pokemon_b = pokemon_df
                if search_term_b:
                    filtered_pokemon_b = pokemon_df[pokemon_df['Name'].str.lower().str.contains(search_term_b.lower())]
                
                pokemon_b_id = st.selectbox(
                    'Selecciona el segundo Pokémon',
                    options=filtered_pokemon_b['ID'].tolist(),
                    format_func=lambda x: filtered_pokemon_b[filtered_pokemon_b['ID'] == x]['Name'].iloc[0] if x in filtered_pokemon_b['ID'].values else x,
                    key='p2'
                )
            
            # Preparar datos para la tarjeta de Pokémon
            pokemon_a = pokemon_df[pokemon_df['ID'] == pokemon_a_id].iloc[0]
            pokemon_b = pokemon_df[pokemon_df['ID'] == pokemon_b_id].iloc[0]
            
            # Mostrar tarjetas de Pokémon
            col1, col2 = st.columns(2)
            
            with col1:
                stats_a = {
                    'HP': pokemon_a['HP'],
                    'Attack': pokemon_a['Attack'],
                    'Defense': pokemon_a['Defense'],
                    'Sp. Atk': pokemon_a['Special Attack'],
                    'Sp. Def': pokemon_a['Special Defense'],
                    'Speed': pokemon_a['Speed']
                }
                show_pokemon_card(
                    pokemon_a['Name'],
                    pokemon_a['SpriteURL'],
                    pokemon_a['Type1'],
                    pokemon_a['Type2'] if pd.notna(pokemon_a['Type2']) else None,
                    stats_a
                )
            
            with col2:
                stats_b = {
                    'HP': pokemon_b['HP'],
                    'Attack': pokemon_b['Attack'],
                    'Defense': pokemon_b['Defense'],
                    'Sp. Atk': pokemon_b['Special Attack'],
                    'Sp. Def': pokemon_b['Special Defense'],
                    'Speed': pokemon_b['Speed']
                }
                show_pokemon_card(
                    pokemon_b['Name'],
                    pokemon_b['SpriteURL'],
                    pokemon_b['Type1'],
                    pokemon_b['Type2'] if pd.notna(pokemon_b['Type2']) else None,
                    stats_b
                )
            
            # Mostrar efectividad de tipo
            if st.checkbox("Mostrar efectividad de tipos", True):
                a_type1 = pokemon_a['Type1']
                a_type2 = pokemon_a['Type2'] if pd.notna(pokemon_a['Type2']) else None
                b_type1 = pokemon_b['Type1']
                b_type2 = pokemon_b['Type2'] if pd.notna(pokemon_b['Type2']) else None
                
                a_vs_b_text, b_vs_a_text, a_vs_b_multiplier, b_vs_a_multiplier = generate_type_effectiveness_chart(
                    a_type1, a_type2, b_type1, b_type2
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**{pokemon_a['Name']}** contra **{pokemon_b['Name']}**: {a_vs_b_text}")
                with col2:
                    st.info(f"**{pokemon_b['Name']}** contra **{pokemon_a['Name']}**: {b_vs_a_text}")
                
                # Crear diagrama de colores para efectividad
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))
                
                # Efectividad de A hacia B
                cmap1 = plt.cm.RdYlGn
                ax1.barh(0, a_vs_b_multiplier, color=cmap1(a_vs_b_multiplier/4), height=0.5)
                ax1.set_xlim(0, max(4, a_vs_b_multiplier + 0.5))
                ax1.set_title(f"{pokemon_a['Name']} → {pokemon_b['Name']}")
                ax1.set_yticks([])
                ax1.set_xticks([0, 1, 2, 4])
                ax1.text(a_vs_b_multiplier/2, 0, f"x{a_vs_b_multiplier}", 
                         ha='center', va='center', color='black', fontweight='bold')
                
                # Efectividad de B hacia A
                ax2.barh(0, b_vs_a_multiplier, color=cmap1(b_vs_a_multiplier/4), height=0.5)
                ax2.set_xlim(0, max(4, b_vs_a_multiplier + 0.5))
                ax2.set_title(f"{pokemon_b['Name']} → {pokemon_a['Name']}")
                ax2.set_yticks([])
                ax2.set_xticks([0, 1, 2, 4])
                ax2.text(b_vs_a_multiplier/2, 0, f"x{b_vs_a_multiplier}", 
                         ha='center', va='center', color='black', fontweight='bold')
                
                plt.tight_layout()
                # Guardar plot de efectividad de tipos
                save_streamlit_plot(fig, f"type_effectiveness_{pokemon_a['Name']}_vs_{pokemon_b['Name']}.png", "type_effectiveness")
                st.pyplot(fig)
            
            # Resultados históricos si existen
            a_wins, b_wins, total_matches = data_manager.get_historical_results(pokemon_a_id, pokemon_b_id)
            if total_matches > 0:
                st.subheader("Historial de Combates")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Victorias de " + pokemon_a['Name'], a_wins)
                with col2:
                    st.metric("Victorias de " + pokemon_b['Name'], b_wins)
                with col3:
                    win_rate = round(a_wins / total_matches * 100, 1) if total_matches > 0 else 0
                    st.metric("Tasa de Victoria de " + pokemon_a['Name'], f"{win_rate}%")
                
                # Gráfico de pastel para victorias
                if total_matches > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie([a_wins, b_wins], labels=[pokemon_a['Name'], pokemon_b['Name']], 
                           autopct='%1.1f%%', colors=['#5B8CFF', '#FF5B5B'], startangle=90)
                    ax.axis('equal')
                    ax.set_title("Distribución de Victorias")
                    # Guardar plot de historial
                    save_streamlit_plot(fig, f"victory_distribution_{pokemon_a['Name']}_vs_{pokemon_b['Name']}.png", "historical_results")
                    st.pyplot(fig)
            
            # Botón de predicción
            if st.button('Predecir Ganador', key='predict_button'):
                with st.spinner("Analizando combate..."):
                    # Preparar datos para predicción usando el data manager
                    X_new, metadata = data_manager.prepare_prediction_data(pokemon_a_id, pokemon_b_id)
                    
                    # Hacer predicción usando el model manager
                    prediction, proba = model_manager.predict(X_new)
                    
                    # Determinamos el ganador
                    if proba > 0.5:
                        winner_name = pokemon_a['Name']
                        conf = proba
                    else:
                        winner_name = pokemon_b['Name']
                        conf = 1 - proba
                    
                    # Mostrar resultado
                    if conf > 0.75:
                        confidence_text = "con alta confianza"
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: rgba(25, 135, 84, 0.15); border-left: 4px solid #198754; color: #0d6832;">
                            <h2>🏆 Predicción: <b>{winner_name}</b> ganará {confidence_text} ({conf:.2%})</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    elif conf > 0.6:
                        confidence_text = "probablemente"
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: rgba(13, 110, 253, 0.15); border-left: 4px solid #0d6efd; color: #00429d;">
                            <h2>🏆 Predicción: <b>{winner_name}</b> ganará {confidence_text} ({conf:.2%})</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        confidence_text = "por un pequeño margen"
                        st.markdown(f"""
                        <div class="prediction-box" style="background-color: rgba(255, 193, 7, 0.15); border-left: 4px solid #ffc107; color: #997404;">
                            <h2>🏆 Predicción: <b>{winner_name}</b> podría ganar {confidence_text} ({conf:.2%})</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mostrar probabilidad del combate
                    fig = plot_match_probability(proba, pokemon_a['Name'], pokemon_b['Name'])
                    st.pyplot(fig)
                    
                    # Análisis de factores
                    st.markdown("### Análisis de Factores")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ventajas estadísticas
                        st.subheader("Comparación de Estadísticas")
                        advantages, total_advantage = calculate_stat_advantage(stats_a, stats_b)
                        
                        # Crear DataFrame para mostrar ventajas
                        adv_df = pd.DataFrame({
                            'Estadística': list(advantages.keys()),
                            'Diferencia': list(advantages.values())
                        })
                        
                        # Crear gráfico de barras horizontal
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.barh(adv_df['Estadística'], adv_df['Diferencia'])
                        
                        # Colorear barras basadas en valores
                        for i, bar in enumerate(bars):
                            if adv_df['Diferencia'].iloc[i] > 0:
                                bar.set_color('#5B8CFF')  # Azul para Pokemon A
                            else:
                                bar.set_color('#FF5B5B')  # Rojo para Pokemon B
                        
                        # Añadir línea vertical en 0
                        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        
                        # Configuración adicional
                        ax.set_title('Diferencia en Estadísticas')
                        ax.set_xlabel('Ventaja de A → | ← Ventaja de B')
                        
                        # Añadir etiquetas con valores
                        for i, v_val in enumerate(adv_df['Diferencia']):
                            if v_val > 0:
                                ax.text(v_val + 2, i, f"+{v_val}", va='center')
                            else:
                                ax.text(v_val - 15, i, str(v_val), va='center')
                        
                        # Guardar plot de comparación de estadísticas
                        save_streamlit_plot(fig, f"stat_comparison_{pokemon_a['Name']}_vs_{pokemon_b['Name']}.png", "stat_comparisons")
                        st.pyplot(fig)
                    
                    with col2:
                        # Explicación SHAP
                        st.subheader("Factores Determinantes")
                        try:
                            fig = explain_prediction_shap(model_manager.model, X_new, get_model_feature_columns())
                            # Guardar plot SHAP
                            save_streamlit_plot(fig, f"shap_explanation_{pokemon_a['Name']}_vs_{pokemon_b['Name']}.png", "shap_explanations")
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"No se pudo generar la explicación SHAP: {e}")
                            
                            # Alternativa: Mostrar importancias de características
                            fig = plot_feature_importance(model_manager.model, get_model_feature_columns())
                            st.pyplot(fig)
                    
                    # Insights adicionales
                    st.markdown("### Insights Adicionales")
                    insights = []
                    
                    # Insight de velocidad
                    speed_diff = stats_a['Speed'] - stats_b['Speed']
                    if abs(speed_diff) > 20:
                        faster = pokemon_a['Name'] if speed_diff > 0 else pokemon_b['Name']
                        insights.append(f"**{faster}** tiene una ventaja significativa en velocidad, lo que le permite atacar primero.")
                    
                    # Insight de tipo
                    if a_vs_b_multiplier > 1.5 or b_vs_a_multiplier > 1.5:
                        advantage_pokemon_name = f"**{pokemon_a['Name']}**" if a_vs_b_multiplier > b_vs_a_multiplier else f"**{pokemon_b['Name']}**"
                        insights.append(f"{advantage_pokemon_name} tiene una fuerte ventaja de tipo en este combate.")
                    
                    # Insight de estadísticas
                    total_stats_a = sum(stats_a.values())
                    total_stats_b = sum(stats_b.values())
                    if abs(total_stats_a - total_stats_b) > 50:
                        stronger = f"**{pokemon_a['Name']}**" if total_stats_a > total_stats_b else f"**{pokemon_b['Name']}**"
                        insights.append(f"{stronger} tiene estadísticas base totales superiores ({total_stats_a if total_stats_a > total_stats_b else total_stats_b} vs {total_stats_b if total_stats_a > total_stats_b else total_stats_a}).")
                    
                    # Insight basado en historial
                    if total_matches > 5:
                        dominant = f"**{pokemon_a['Name']}**" if a_wins > b_wins else f"**{pokemon_b['Name']}**"
                        insights.append(f"{dominant} ha ganado históricamente más veces en este enfrentamiento ({a_wins if a_wins > b_wins else b_wins} de {total_matches} combates).")
                    
                    if insights:
                        for insight in insights:
                            st.markdown(f"- {insight}")
                    else:
                        st.markdown("No hay insights adicionales destacables para este combate.")
        
        with tab2:
            st.header("Análisis del Modelo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Características Importantes")
                fig = plot_feature_importance(model_manager.model, get_model_feature_columns())
                st.pyplot(fig)
            
            with col2:
                st.subheader("Distribución de Pokémon por Tipo Primario")
                
                type_counts = pokemon_df['Type1'].value_counts()
                type_wins = pd.DataFrame({
                    'Tipo': type_counts.index,
                    'Cantidad': type_counts.values
                })
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(type_wins['Tipo'], type_wins['Cantidad'])
                
                for i, bar in enumerate(bars):
                    tipo = type_wins['Tipo'].iloc[i]
                    bar.set_color(TYPE_COLORS.get(tipo, '#777'))
                
                plt.xticks(rotation=45, ha='right')
                plt.title('Distribución de Pokémon por Tipo Primario')
                plt.tight_layout()
                
                # Guardar plot de distribución de tipos
                save_streamlit_plot(fig, "pokemon_type_distribution.png", "pokemon_analysis")
                st.pyplot(fig)
            
            st.subheader("Rendimiento del Modelo")
            
            X = features_df.drop(['pokemon_A_id', 'pokemon_B_id', 'winner_id', 'A_Name', 'B_Name', 
                               'A_SpriteURL', 'B_SpriteURL', 'target'], axis=1, errors='ignore')
            y = features_df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model_manager.model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{acc:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{prec:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{rec:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1-Score", f"{f1:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Valor Real')
            ax.set_title('Matriz de Confusión')
            ax.set_xticklabels(['Pierde A', 'Gana A'])
            ax.set_yticklabels(['Pierde A', 'Gana A'])
            # Guardar matriz de confusión
            save_streamlit_plot(fig, "confusion_matrix.png", "model_performance")
            st.pyplot(fig)
            
            st.subheader("Reporte de Clasificación")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        with tab3:
            st.header("Acerca de este Predictor")
            
            st.markdown("""
            ### Sistema de Predicción de Combates Pokémon
            
            Este sistema utiliza aprendizaje automático avanzado para predecir el resultado más probable
            de un enfrentamiento 1vs1 entre dos Pokémon, basándose en datos reales de combates.
            
            #### Características principales:
            
            - **Modelo XGBoost**: Algoritmo de gradient boosting que destaca en problemas de clasificación
            - **Análisis de tipos**: Considera las ventajas/desventajas de tipo según la tabla oficial
            - **Datos reales**: Entrenado con combates reales, no simulados
            - **Explicabilidad**: Utiliza SHAP para explicar las predicciones
            
            #### Factores considerados para la predicción:
            
            - Estadísticas base de cada Pokémon (HP, Ataque, Defensa, etc.)
            - Tipos y sus interacciones (ventajas/desventajas)
            - Diferencias relativas entre estadísticas
            - Patrones históricos de victorias/derrotas
            
            #### Limitaciones actuales:
            
            - No considera movimientos específicos de cada Pokémon
            - No tiene en cuenta habilidades especiales
            - No incorpora estrategias de combate o items
            
            #### Próximas mejoras:
            
            - Incorporación de movimientos y habilidades
            - Análisis de estrategias comunes
            - Mejora de visualizaciones e insights
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Estadísticas")
                stats_analysis = pokemon_df[['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']]
                fig, ax = plt.subplots(figsize=(8, 6))
                stats_analysis.boxplot(ax=ax)
                plt.xticks(rotation=45)
                plt.title('Distribución de Estadísticas Base')
                plt.tight_layout()
                # Guardar distribución de estadísticas
                save_streamlit_plot(fig, "stats_distribution.png", "pokemon_analysis")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Matriz de Correlación")
                corr = stats_analysis.corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                plt.title('Correlación entre Estadísticas')
                plt.tight_layout()
                # Guardar matriz de correlación
                save_streamlit_plot(fig, "stats_correlation.png", "pokemon_analysis")
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error al cargar los datos o entrenar el modelo: {e}")
        st.error("Asegúrate de que los archivos CSV estén disponibles en el directorio correcto.")

if __name__ == "__main__":
    main()