# 🎮 Predictor de Combates Pokémon

## 📋 Descripción del Proyecto

Sistema de predicción avanzado que utiliza machine learning para predecir el resultado de combates 1vs1 entre Pokémon, basándose en datos reales de combates, estadísticas base y efectividad de tipos.

## ✨ Características Principales

- 🤖 **Modelo XGBoost** optimizado para predicción de combates
- 📊 **Análisis SHAP** para explicabilidad de predicciones  
- 🎨 **Interfaz web interactiva** con Streamlit
- 📈 **Visualizaciones avanzadas** de estadísticas y probabilidades
- 🔍 **Búsqueda inteligente** de Pokémon con filtros
- 📚 **Historial de combates** y análisis de tendencias
- ⚡ **Análisis de efectividad** de tipos en tiempo real
- 💾 **Guardado automático de gráficas** en carpetas organizadas
- 🎯 **Colores corregidos** en gráficas de probabilidad
- 📁 **Gestión inteligente de plots** con metadatos y timestamps

## 🚀 Instalación Rápida

### Opción 1: Script Automático (Recomendado)
```fish
./setup.fish
./start_app.fish
```

### Opción 2: Instalación Manual
```fish
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate.fish

# Instalar dependencias
pip install -r requirements.txt

# Iniciar aplicación
streamlit run streamlit_app.py
```

## 📁 Estructura del Proyecto

```
├── 📄 streamlit_app.py          # Aplicación principal
├── 📄 data_utils.py             # Utilidades de datos
├── 📄 advanced_data_manager.py  # Gestión avanzada de datos
├── 📄 ui_utils.py               # Utilidades de interfaz
├── 📄 plot_utils.py             # Funciones de visualización
├── 📄 shap_utils.py             # Análisis SHAP
├── 📄 config.py                 # Configuración global
├── 📄 utils.py                  # Funciones auxiliares
├── 📄 requirements.txt          # Dependencias Python
├── 📄 setup.fish                # Script de instalación
├── 📁 data/                     # Datasets
│   ├── all_pokemon_data.csv     # Datos de Pokémon
│   ├── combats.csv              # Datos de combates
│   └── combats_features.csv     # Características extraídas
├── 📁 tests/                    # Tests unitarios
├── 📁 .streamlit/               # Configuración de Streamlit
└── 📁 models/                   # Modelos entrenados
```

## 🔧 Uso de la Aplicación

### 1. Predicción de Combates
- Selecciona dos Pokémon usando la búsqueda inteligente
- Filtra por tipo, generación o estadísticas
- Obtén predicción con nivel de confianza
- Analiza factores determinantes con SHAP

### 2. Análisis del Modelo
- Visualiza importancia de características
- Examina matriz de confusión
- Revisa métricas de rendimiento
- Explora distribuciones de tipos

### 3. Información Detallada
- Consulta efectividad de tipos
- Revisa historial de combates
- Analiza ventajas estadísticas
- Obtén insights inteligentes

## 📊 Factores de Predicción

El modelo considera múltiples factores:

### Estadísticas Base
- **HP**: Puntos de vida
- **Attack**: Ataque físico
- **Defense**: Defensa física  
- **Special Attack**: Ataque especial
- **Special Defense**: Defensa especial
- **Speed**: Velocidad

### Análisis de Tipos
- Ventajas/desventajas según tabla oficial
- Efectividad de tipo primario y secundario
- Multiplicadores de daño

### Diferencias Relativas
- Comparación directa de estadísticas
- Ratios de fortaleza/debilidad
- Análisis de roles de combate

### Datos Históricos
- Resultados de combates previos
- Patrones de victoria/derrota
- Tendencias por tipo

## 📈 Gestión Automática de Gráficas

### 🗂️ Organización Inteligente
Todas las gráficas generadas se guardan automáticamente en la carpeta `plots/` con una estructura organizada:

```
plots/
├── match_predictions/     # Predicciones de combates individuales
├── type_effectiveness/    # Análisis de efectividad de tipos
├── historical_results/    # Resultados históricos de combates  
├── stat_comparisons/      # Comparaciones de estadísticas
├── shap_explanations/     # Explicaciones SHAP del modelo
├── model_analysis/        # Análisis del modelo (importancia, etc.)
├── model_performance/     # Métricas de rendimiento del modelo
├── pokemon_analysis/      # Análisis de datos de Pokémon
├── eda_plots/            # Análisis exploratorio de datos
└── custom/               # Gráficos personalizados
```

### 🕒 Características del Sistema de Plots
- **Timestamps automáticos**: Evita sobreescribir archivos
- **Metadatos enriquecidos**: Cada gráfica incluye información contextual
- **Registro JSON**: Historial completo de todas las gráficas generadas
- **Limpieza automática**: Elimina gráficas antiguas según configuración
- **Informes de resumen**: Estadísticas de uso y categorización

### 🎨 Mejoras Visuales Implementadas
- ✅ **Colores corregidos** en gráficas de probabilidad de combate
- ✅ **Posicionamiento correcto** de nombres de Pokémon
- ✅ **Guardado automático** de todas las visualizaciones
- ✅ **Organización por categorías** temáticas
- ✅ **Alta resolución** (300 DPI) para todas las gráficas

## 🧪 Testing

Ejecutar tests unitarios:
```fish
./run_tests.fish
```

O manualmente:
```fish
python -m pytest tests/ -v
```

## 📈 Métricas del Modelo

- **Accuracy**: 81.12%
- **Precision**: 80.95%
- **Recall**: 77.88%
- **F1-Score**: 79.39%

*Evaluado sobre 9,610 combates del conjunto de prueba (20% del dataset total)*

## 🛠️ Configuración Avanzada

### Variables de Entorno
```fish
export POKEMON_MODEL_PATH="custom/path/model.pkl"
export POKEMON_DATA_DIR="custom/data/directory"
export STREAMLIT_SERVER_PORT=8502
```

### Configuración de Streamlit
Editar `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF5B5B"
backgroundColor = "#FFFFFF"

[server]
maxUploadSize = 200
```

## 🔧 Personalización

### Añadir Nuevos Tipos de Análisis
1. Editar `plot_utils.py` para nuevas visualizaciones
2. Agregar funciones en `utils.py` para análisis
3. Integrar en `streamlit_app.py`

### Modificar Modelo
1. Editar `advanced_data_manager.py`
2. Ajustar hiperparámetros en configuración
3. Re-entrenar con nuevos datos

### Personalizar Interfaz
1. Modificar estilos CSS en `ui_utils.py`
2. Agregar nuevos componentes
3. Actualizar configuración de tema

## 🐛 Solución de Problemas

### Error: Modelo no encontrado
```fish
# Forzar reentrenamiento
rm pokemon_battle_model_xgboost.pkl
python -c "import streamlit_app; streamlit_app.main()"
```

### Error: Datos faltantes
```fish
# Verificar archivos de datos
ls -la data/
# Regenerar datos si es necesario
python clean_data.py
```

### Error: Dependencias
```fish
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

## 📚 Recursos Adicionales

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Pokémon Type Chart](https://pokemondb.net/type)

## 🤝 Contribución

1. Fork el repositorio
2. Crea una branch para tu feature
3. Implementa cambios con tests
4. Envía pull request

### Estándares de Código
- Seguir PEP 8 para Python
- Documentar funciones con docstrings
- Mantener cobertura de tests >80%
- Usar type hints cuando sea posible

## 📄 Licencia

MIT License - ver archivo LICENSE para detalles

## 👥 Créditos

- Datos de Pokémon: [PokéAPI](https://pokeapi.co/)
- Algoritmo: XGBoost, scikit-learn
- Interfaz: Streamlit
- Explicabilidad: SHAP

## 🔄 Changelog

### v1.0.0
- ✅ Predicción básica de combates
- ✅ Interfaz web con Streamlit
- ✅ Análisis SHAP básico

### v1.1.0 (Actual)
- ✅ Gestión avanzada de datos
- ✅ Tests unitarios
- ✅ Script de instalación automatizado
- ✅ Mejoras en la interfaz
- ✅ Documentación completa

### Próximas versiones
- 🔮 Análisis de movimientos específicos
- 🔮 Predicción de equipos completos
- 🔮 API REST para integración
- 🔮 Modo multijugador

---

**¿Listo para predecir el próximo campeón? ¡Ejecuta `./setup.fish` y comienza!** 🏆
