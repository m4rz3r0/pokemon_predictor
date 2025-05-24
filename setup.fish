#!/usr/bin/env fish

# Script de instalación y configuración para el Predictor de Combates Pokémon
# Uso: ./setup.fish

echo "🎮 Configurando Predictor de Combates Pokémon..."

# Verificar que estamos en el directorio correcto
if not test -f "streamlit_app.py"
    echo "❌ Error: Ejecutar desde el directorio raíz del proyecto"
    exit 1
end

# Función para verificar comando
function check_command
    if not command -v $argv[1] >/dev/null 2>&1
        echo "❌ Error: $argv[1] no está instalado"
        return 1
    else
        echo "✅ $argv[1] encontrado"
        return 0
    end
end

# Verificar dependencias del sistema
echo "🔍 Verificando dependencias del sistema..."
check_command python3; or exit 1
check_command pip3; or exit 1

# Verificar versión de Python
set python_version (python3 --version | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

if test (echo $python_version | cut -d'.' -f1) -lt 3; or test (echo $python_version | cut -d'.' -f2) -lt 8
    echo "❌ Error: Se requiere Python 3.8 o superior"
    exit 1
end

# Crear entorno virtual si no existe
if not test -d "venv"
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
else
    echo "📦 Entorno virtual ya existe"
end

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate.fish

# Actualizar pip
echo "⬆️ Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📥 Instalando dependencias..."
if test -f "requirements.txt"
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt no encontrado"
    exit 1
end

# Verificar instalación de paquetes críticos
echo "🔍 Verificando instalaciones..."
set critical_packages streamlit pandas numpy sklearn xgboost matplotlib seaborn

for package in $critical_packages
    if python3 -c "import $package" 2>/dev/null
        echo "✅ $package instalado correctamente"
    else
        echo "❌ Error: $package no se instaló correctamente"
        exit 1
    end
end

# Verificar estructura de datos
echo "📂 Verificando estructura de datos..."
set required_files "data/all_pokemon_data.csv" "data/combats.csv" "data/combats_features.csv"

for file in $required_files
    if test -f $file
        echo "✅ $file encontrado"
    else
        echo "⚠️ Advertencia: $file no encontrado"
        echo "   El archivo se puede generar ejecutando los scripts de preparación de datos"
    end
end

# Crear directorios necesarios
echo "📁 Creando directorios necesarios..."
mkdir -p logs
mkdir -p models
mkdir -p plots
echo "✅ Directorios creados"

# Verificar modelo entrenado
echo "🤖 Verificando modelo..."
if test -f "pokemon_battle_model_xgboost.pkl"
    echo "✅ Modelo entrenado encontrado"
else
    echo "⚠️ Advertencia: Modelo no encontrado"
    echo "   El modelo se entrenará automáticamente en el primer uso"
end

# Configurar Git hooks (opcional)
if test -d ".git"
    echo "🔗 Configurando Git hooks..."
    # Crear pre-commit hook para ejecutar tests
    echo "#!/bin/sh" > .git/hooks/pre-commit
    echo "python -m pytest tests/ --tb=short" >> .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "✅ Git hooks configurados"
end

# Crear script de inicio
echo "🚀 Creando script de inicio..."
echo "#!/usr/bin/env fish" > start_app.fish
echo "source venv/bin/activate.fish" >> start_app.fish
echo "streamlit run streamlit_app.py" >> start_app.fish
chmod +x start_app.fish
echo "✅ Script de inicio creado: ./start_app.fish"

# Crear script de tests
echo "📝 Creando script de tests..."
echo "#!/usr/bin/env fish" > run_tests.fish
echo "source venv/bin/activate.fish" >> run_tests.fish
echo "python -m pytest tests/ -v --tb=short" >> run_tests.fish
chmod +x run_tests.fish
echo "✅ Script de tests creado: ./run_tests.fish"

# Configuración completada
echo "🎉 ¡Configuración completada!"
echo ""
echo "📋 Próximos pasos:"
echo "  1. Activar entorno virtual: source venv/bin/activate.fish"
echo "  2. Iniciar aplicación: ./start_app.fish"
echo "  3. Ejecutar tests: ./run_tests.fish"
echo ""
echo "🌐 La aplicación estará disponible en: http://localhost:8501"
echo ""
echo "📚 Documentación adicional en: mejoras_propuestas.md"

# Mostrar información del sistema
echo "📊 Información del sistema:"
echo "  • Python: $python_version"
echo "  • Sistema: "(uname -s)
echo "  • Directorio: "(pwd)
echo "  • Espacio disponible: "(df -h . | tail -1 | awk '{print $4}')

echo "✨ ¡Listo para predecir combates Pokémon!"
