"""
Tests unitarios para el sistema de predicción de combates Pokémon
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

# Importar las clases a testear
try:
    from advanced_data_manager import PokemonDataManager, PokemonModelManager
    from data_utils import prepare_prediction_data, get_historical_results
    from utils import generate_battle_insights, format_stat_comparison
except ImportError as e:
    pytest.skip(f"No se pudieron importar módulos: {e}", allow_module_level=True)

class TestPokemonDataManager:
    """Tests para la gestión de datos"""
    
    @pytest.fixture
    def sample_pokemon_data(self):
        """Datos de ejemplo para testing"""
        return pd.DataFrame({
            'ID': ['1', '2', '3'],
            'Name': ['Bulbasaur', 'Ivysaur', 'Venusaur'],
            'HP': [45, 60, 80],
            'Attack': [49, 62, 82],
            'Defense': [49, 63, 83],
            'Special Attack': [65, 80, 100],
            'Special Defense': [65, 80, 100],
            'Speed': [45, 60, 80],
            'Type1': ['Grass', 'Grass', 'Grass'],
            'Type2': ['Poison', 'Poison', 'Poison'],
            'SpriteURL': ['url1', 'url2', 'url3']
        })
    
    @pytest.fixture
    def sample_combat_data(self):
        """Datos de combate de ejemplo"""
        return pd.DataFrame({
            'First_pokemon': ['1', '2', '1'],
            'Second_pokemon': ['2', '3', '3'],
            'Winner': ['2', '3', '1']
        })
    
    @pytest.fixture
    def temp_data_dir(self, sample_pokemon_data, sample_combat_data):
        """Directorio temporal con datos de prueba"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Guardar datos de prueba
            pokemon_file = os.path.join(tmpdir, 'all_pokemon_data.csv')
            combat_file = os.path.join(tmpdir, 'combats.csv')
            
            sample_pokemon_data.to_csv(pokemon_file, index=False)
            sample_combat_data.to_csv(combat_file, index=False)
            
            yield tmpdir
    
    def test_pokemon_data_loading(self, temp_data_dir):
        """Test de carga de datos de Pokémon"""
        manager = PokemonDataManager(data_dir=temp_data_dir)
        
        # Test carga exitosa
        pokemon_data = manager.pokemon_data
        assert not pokemon_data.empty
        assert 'Bulbasaur' in pokemon_data['Name'].values
        assert pokemon_data['ID'].dtype == 'object'  # String IDs
    
    def test_pokemon_search(self, temp_data_dir):
        """Test de búsqueda de Pokémon"""
        manager = PokemonDataManager(data_dir=temp_data_dir)
        
        # Búsqueda por nombre
        results = manager.search_pokemon("bulba")
        assert len(results) == 1
        assert results.iloc[0]['Name'] == 'Bulbasaur'
        
        # Búsqueda por tipo
        results = manager.search_pokemon(type_filter="Grass")
        assert len(results) == 3  # Todos son tipo Grass
    
    def test_get_pokemon_by_id(self, temp_data_dir):
        """Test de obtención de Pokémon por ID"""
        manager = PokemonDataManager(data_dir=temp_data_dir)
        
        # ID válido
        pokemon = manager.get_pokemon_by_id('1')
        assert pokemon['Name'] == 'Bulbasaur'
        
        # ID inválido
        with pytest.raises(ValueError):
            manager.get_pokemon_by_id('999')
    
    def test_historical_results(self, temp_data_dir):
        """Test de resultados históricos"""
        manager = PokemonDataManager(data_dir=temp_data_dir)
        
        # Combate con historial
        a_wins, b_wins, total = manager.get_historical_results('1', '2')
        assert total > 0
        assert a_wins + b_wins == total
    
    def test_validation_errors(self):
        """Test de errores de validación"""
        # Directorio inexistente
        with pytest.raises(FileNotFoundError):
            manager = PokemonDataManager(data_dir="/directorio/inexistente")
            _ = manager.pokemon_data

class TestUtilityFunctions:
    """Tests para funciones utilitarias"""
    
    def test_format_stat_comparison(self):
        """Test de comparación de estadísticas"""
        stats_a = {'HP': 100, 'Attack': 80, 'Speed': 90}
        stats_b = {'HP': 80, 'Attack': 100, 'Speed': 90}
        
        comparison = format_stat_comparison(stats_a, stats_b, "PokemonA", "PokemonB")
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert comparison[comparison['Estadística'] == 'HP']['Ventaja'].iloc[0] == "PokemonA"
        assert comparison[comparison['Estadística'] == 'Attack']['Ventaja'].iloc[0] == "PokemonB"
        assert comparison[comparison['Estadística'] == 'Speed']['Ventaja'].iloc[0] == "Empate"
    
    def test_generate_battle_insights(self):
        """Test de generación de insights"""
        # Datos de prueba
        pokemon_a = pd.Series({
            'Name': 'Charizard',
            'HP': 78, 'Attack': 84, 'Defense': 78,
            'Special Attack': 109, 'Special Defense': 85, 'Speed': 100
        })
        
        pokemon_b = pd.Series({
            'Name': 'Blastoise',
            'HP': 79, 'Attack': 83, 'Defense': 100,
            'Special Attack': 85, 'Special Defense': 105, 'Speed': 78
        })
        
        type_effectiveness = (1.0, 2.0)  # B tiene ventaja sobre A
        historical_data = (3, 7, 10)  # B ha ganado 7 de 10
        
        insights = generate_battle_insights(
            pokemon_a, pokemon_b, type_effectiveness, historical_data
        )
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert any("ventaja de tipo" in insight for insight in insights)
        assert any("históricamente" in insight for insight in insights)

class TestModelIntegration:
    """Tests de integración del modelo"""
    
    @patch('advanced_data_manager.joblib.load')
    def test_model_loading(self, mock_load):
        """Test de carga del modelo"""
        # Mock del modelo
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_load.return_value = mock_model
        
        model_manager = PokemonModelManager('fake_model.pkl')
        
        # Test predicción
        X = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        prediction, probability = model_manager.predict(X)
        
        assert prediction == 1
        assert probability == 0.7

class TestDataIntegrity:
    """Tests de integridad de datos"""
    
    def test_data_consistency(self):
        """Test de consistencia entre datasets"""
        # Este test verificaría que:
        # - Todos los Pokémon en combats.csv existen en all_pokemon_data.csv
        # - Los IDs son consistentes entre archivos
        # - No hay valores faltantes críticos
        pass  # Implementar cuando tengamos acceso a datos reales
    
    def test_feature_completeness(self):
        """Test de completitud de características"""
        # Verificar que todas las características necesarias están presentes
        pass  # Implementar según características específicas del modelo

if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"])
