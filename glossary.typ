// =============================================
// GLOSARIO DE TÉRMINOS - SISTEMA DE PREDICCIÓN POKÉMON
// =============================================
// Este archivo contiene todas las definiciones de términos técnicos
// y conceptos específicos del dominio utilizados en la documentación.

#let glossary-entries = (
  // Términos de Machine Learning
  (
    key: "accuracy",
    short: "Accuracy",
    description: "Métrica que mide la proporción de predicciones correctas sobre el total de predicciones realizadas. Se calcula como (VP + VN) / (VP + VN + FP + FN).",
    group: "Machine Learning"
  ),
  (
    key: "clasificacion-binaria",
    short: "Clasificación Binaria",
    description: "Tipo de problema de machine learning donde se debe asignar cada instancia a una de dos clases posibles (en este caso: \"Pokémon A gana\" o \"Pokémon B gana\").",
    group: "Machine Learning"
  ),
  (
    key: "ensemble-learning",
    short: "Ensemble Learning",
    description: "Técnica que combina múltiples modelos de aprendizaje para crear un sistema de predicción más robusto y preciso que cualquier modelo individual.",
    group: "Machine Learning"
  ),
  (
    key: "f1-score",
    short: "F1-Score",
    description: "Media armónica de precision y recall, proporcionando una métrica balanceada especialmente útil en conjuntos de datos desbalanceados. Se calcula como 2 × (Precision × Recall) / (Precision + Recall).",
    group: "Machine Learning"
  ),
  (
    key: "feature-engineering",
    short: "Feature Engineering",
    description: "Proceso de crear, seleccionar y transformar variables de entrada para mejorar el rendimiento del modelo de machine learning.",
    group: "Machine Learning"
  ),
  (
    key: "gradient-boosting",
    short: "Gradient Boosting",
    description: "Técnica de ensemble que construye modelos secuencialmente, donde cada nuevo modelo corrige los errores del conjunto anterior.",
    group: "Machine Learning"
  ),
  (
    key: "one-hot-encoding",
    short: "One-Hot Encoding",
    description: "Técnica de codificación que convierte variables categóricas en vectores binarios, creando una columna por cada categoría posible.",
    group: "Machine Learning"
  ),
  (
    key: "overfitting",
    short: "Overfitting",
    description: "Fenómeno donde un modelo aprende demasiado específicamente los datos de entrenamiento, perdiendo capacidad de generalización a datos nuevos.",
    group: "Machine Learning"
  ),
  (
    key: "pipeline",
    short: "Pipeline",
    description: "Secuencia de pasos de procesamiento de datos y modelado que se pueden ejecutar de manera coordinada y reproducible.",
    group: "Machine Learning"
  ),
  (
    key: "precision",
    short: "Precision",
    description: "Métrica que mide la proporción de predicciones positivas que fueron correctas. Se calcula como VP / (VP + FP).",
    group: "Machine Learning"
  ),
  (
    key: "recall",
    short: "Recall",
    description: "Métrica que mide la proporción de casos positivos reales que fueron identificados correctamente. Se calcula como VP / (VP + FN).",
    group: "Machine Learning"
  ),
  (
    key: "regularizacion",
    short: "Regularización",
    description: "Técnicas para prevenir el sobreajuste añadiendo penalizaciones por complejidad al proceso de entrenamiento.",
    group: "Machine Learning"
  ),
  (
    key: "shap",
    short: "SHAP",
    long: "SHapley Additive exPlanations",
    description: "Método para explicar predicciones individuales de modelos de machine learning, basado en la teoría de juegos cooperativos.",
    group: "Machine Learning"
  ),
  (
    key: "train-test-split",
    short: "Train-Test Split",
    description: "División del conjunto de datos en conjuntos separados para entrenamiento y evaluación, asegurando que el modelo sea evaluado en datos no vistos durante el entrenamiento.",
    group: "Machine Learning"
  ),
  (
    key: "xgboost",
    short: "XGBoost",
    description: "Implementación optimizada de gradiente potenciado diseñada para ser escalable, portátil y eficiente.",
    group: "Machine Learning"
  ),

  // Términos del Dominio Pokémon
  (
    key: "estadisticas-base",
    short: "Estadísticas Base",
    description: "Valores fundamentales que definen las capacidades de un Pokémon: HP (Puntos de Salud), Attack (Ataque), Defense (Defensa), Special Attack (Ataque Especial), Special Defense (Defensa Especial), y Speed (Velocidad).",
    group: "Dominio Pokémon"
  ),
  (
    key: "efectividad-tipos",
    short: "Efectividad de Tipos",
    description: "Sistema de ventajas y desventajas entre diferentes tipos elementales (ej: Agua es súper efectivo contra Fuego).",
    group: "Dominio Pokémon"
  ),
  (
    key: "meta-juego",
    short: "Meta-juego",
    description: "Estrategias, tendencias y patrones dominantes en el juego competitivo en un momento dado.",
    group: "Dominio Pokémon"
  ),
  (
    key: "matchup",
    short: "Matchup",
    description: "Enfrentamiento específico entre dos Pokémon o equipos, considerando sus fortalezas y debilidades relativas.",
    group: "Dominio Pokémon"
  ),
  (
    key: "sprite",
    short: "Sprite",
    description: "Imagen gráfica pequeña que representa a un Pokémon en el juego.",
    group: "Dominio Pokémon"
  ),
  (
    key: "tipo-elemental",
    short: "Tipo Elemental",
    description: "Categoría que define las características fundamentales de un Pokémon (ej: Fuego, Agua, Planta, etc.). Los Pokémon pueden tener uno o dos tipos.",
    group: "Dominio Pokémon"
  ),

  // Términos Técnicos del Sistema
  (
    key: "cache",
    short: "Caché",
    description: "Almacenamiento temporal de datos o resultados computacionales para evitar recálculos innecesarios y mejorar el rendimiento.",
    group: "Sistema Técnico"
  ),
  (
    key: "conjunto-datos",
    short: "Conjunto de Datos",
    description: "Conjunto de datos estructurados utilizado para entrenamiento y evaluación del modelo.",
    group: "Sistema Técnico"
  ),
  (
    key: "endpoint",
    short: "Endpoint",
    description: "Punto de acceso específico en una API o aplicación web.",
    group: "Sistema Técnico"
  ),
  (
    key: "feature-importance",
    short: "Feature Importance",
    description: "Medida de la relevancia relativa de cada característica en las decisiones del modelo.",
    group: "Sistema Técnico"
  ),
  (
    key: "force-plot",
    short: "Force Plot",
    description: "Tipo específico de visualización SHAP que muestra cómo cada característica contribuye a empujar una predicción desde un valor base hacia el resultado final.",
    group: "Sistema Técnico"
  ),
  (
    key: "latencia",
    short: "Latencia",
    description: "Tiempo que transcurre entre una solicitud y su respuesta correspondiente.",
    group: "Sistema Técnico"
  ),
  (
    key: "pickle",
    short: "Pickle",
    description: "Formato de serialización de Python utilizado para guardar objetos complejos como modelos entrenados.",
    group: "Sistema Técnico"
  ),
  (
    key: "api",
    short: "API",
    long: "Application Programming Interface",
    description: "Conjunto de definiciones y protocolos que permiten la comunicación entre diferentes componentes de software.",
    group: "Sistema Técnico"
  ),
  (
    key: "dataframe",
    short: "DataFrame",
    description: "Estructura de datos tabular bidimensional de la librería Pandas, similar a una hoja de cálculo.",
    group: "Sistema Técnico"
  ),
  (
    key: "framework",
    short: "Framework",
    description: "Marco de trabajo de Python para crear aplicaciones web interactivas de ciencia de datos de manera rápida.",
    group: "Sistema Técnico"
  ),

  // Términos de Evaluación
  (
    key: "test-set",
    short: "Test Set",
    description: "Subconjunto del dataset reservado exclusivamente para evaluación final del modelo, nunca utilizado durante el entrenamiento.",
    group: "Evaluación"
  ),
  (
    key: "training-set",
    short: "Training Set",
    description: "Subconjunto del dataset utilizado para entrenar el modelo de machine learning.",
    group: "Evaluación"
  ),
  (
    key: "falso-negativo",
    short: "Falso Negativo",
    description: "Caso donde el modelo predice la clase negativa pero la real es positiva.",
    group: "Evaluación"
  ),
  (
    key: "falso-positivo",
    short: "Falso Positivo",
    description: "Caso donde el modelo predice la clase positiva pero la real es negativa.",
    group: "Evaluación"
  ),
  (
    key: "matriz-confusion",
    short: "Matriz de Confusión",
    description: "Tabla que resume el rendimiento de un clasificador mostrando la distribución de predicciones correctas e incorrectas.",
    group: "Evaluación"
  ),
  (
    key: "cross-validation",
    short: "Cross-Validation",
    description: "Técnica que divide el dataset en múltiples pliegues para evaluar la estabilidad y generalización del modelo.",
    group: "Evaluación"
  ),
  (
    key: "verdadero-negativo",
    short: "Verdadero Negativo",
    description: "Caso donde el modelo predice correctamente la clase negativa.",
    group: "Evaluación"
  ),
  (
    key: "verdadero-positivo",
    short: "Verdadero Positivo",
    description: "Caso donde el modelo predice correctamente la clase positiva.",
    group: "Evaluación"
  ),

  // Términos de Arquitectura de Software
  (
    key: "arquitectura-modular",
    short: "Arquitectura Modular",
    description: "Diseño de software que separa funcionalidades en módulos independientes y reutilizables.",
    group: "Arquitectura"
  ),
  (
    key: "capa-presentacion",
    short: "Capa de Presentación",
    description: "Componente de la arquitectura responsable de la interfaz de usuario y la interacción.",
    group: "Arquitectura"
  ),
  (
    key: "carga-perezosa",
    short: "Carga Perezosa",
    long: "Lazy Loading",
    description: "Técnica de optimización que retrasa la carga de recursos hasta que son realmente necesarios.",
    group: "Arquitectura"
  ),
  (
    key: "patron-diseno",
    short: "Patrón de Diseño",
    description: "Solución reutilizable a problemas comunes en el diseño de software.",
    group: "Arquitectura"
  ),
  (
    key: "persistencia",
    short: "Persistencia",
    description: "Almacenamiento duradero de datos que sobrevive al cierre de la aplicación.",
    group: "Arquitectura"
  ),
  (
    key: "escalabilidad",
    short: "Escalabilidad",
    description: "Capacidad de un sistema para manejar cargas de trabajo crecientes manteniendo el rendimiento.",
    group: "Arquitectura"
  ),

  // Términos de Testing y Validación
  (
    key: "benchmark",
    short: "Punto de Referencia",
    description: "Punto de referencia estándar utilizado para comparar el rendimiento de diferentes sistemas o algoritmos.",
    group: "Testing"
  ),
  (
    key: "cobertura-codigo",
    short: "Cobertura de Código",
    description: "Métrica que indica qué porcentaje del código fuente es ejecutado durante las pruebas.",
    group: "Testing"
  ),
  (
    key: "prueba-integracion",
    short: "Prueba de Integración",
    description: "Validación que verifica el funcionamiento correcto de múltiples componentes trabajando juntos.",
    group: "Testing"
  ),
  (
    key: "prueba-unitaria",
    short: "Prueba Unitaria",
    description: "Validación que verifica el funcionamiento correcto de componentes individuales de manera aislada.",
    group: "Testing"
  ),
  (
    key: "regresion",
    short: "Regresión",
    description: "Degradación no intencionada en la funcionalidad debido a cambios en el código.",
    group: "Testing"
  ),
  (
    key: "suite-pruebas",
    short: "Suite de Pruebas",
    description: "Conjunto organizado de pruebas diseñadas para validar diferentes aspectos del sistema.",
    group: "Testing"
  ),
)
