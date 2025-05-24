"""
Gestor de plots para organizar automáticamente las gráficas generadas
"""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

class PlotManager:
    """Clase para gestionar la organización y guardado de plots"""
    
    def __init__(self, base_plots_dir="plots"):
        self.base_plots_dir = Path(base_plots_dir)
        self.base_plots_dir.mkdir(exist_ok=True)
        
        # Estructura de carpetas organizada
        self.subfolders = {
            "match_predictions": "Gráficos de predicciones de combates",
            "type_effectiveness": "Análisis de efectividad de tipos",
            "historical_results": "Resultados históricos de combates",
            "stat_comparisons": "Comparaciones de estadísticas",
            "shap_explanations": "Explicaciones SHAP del modelo",
            "model_analysis": "Análisis del modelo (importancia de características)",
            "model_performance": "Rendimiento del modelo (métricas)",
            "pokemon_analysis": "Análisis de datos de Pokémon",
            "eda_plots": "Análisis exploratorio de datos",
            "custom": "Gráficos personalizados"
        }
        
        # Crear todas las subcarpetas
        self._create_folder_structure()
        
        # Archivo de registro de plots
        self.plot_registry_file = self.base_plots_dir / "plot_registry.json"
        self.plot_registry = self._load_plot_registry()
    
    def _create_folder_structure(self):
        """Crea la estructura de carpetas para organizar los plots"""
        for subfolder, description in self.subfolders.items():
            folder_path = self.base_plots_dir / subfolder
            folder_path.mkdir(exist_ok=True)
            
            # Crear archivo README en cada carpeta
            readme_path = folder_path / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {subfolder.replace('_', ' ').title()}\n\n")
                    f.write(f"{description}\n\n")
                    f.write("Esta carpeta contiene gráficos generados automáticamente por la aplicación.\n")
    
    def _load_plot_registry(self):
        """Carga el registro de plots desde el archivo JSON"""
        if self.plot_registry_file.exists():
            with open(self.plot_registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_plot_registry(self):
        """Guarda el registro de plots en el archivo JSON"""
        with open(self.plot_registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.plot_registry, f, indent=2, ensure_ascii=False)
    
    def save_plot(self, fig, filename, subfolder="custom", metadata=None):
        """
        Guarda un plot en la carpeta correspondiente
        
        Args:
            fig: Objeto figura de matplotlib
            filename: Nombre del archivo (con o sin extensión .png)
            subfolder: Subcarpeta donde guardar (debe estar en self.subfolders)
            metadata: Diccionario con metadatos adicionales
        
        Returns:
            str: Ruta completa del archivo guardado
        """
        # Validar subfolder
        if subfolder not in self.subfolders:
            subfolder = "custom"
        
        # Asegurar extensión .png
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        
        # Añadir timestamp para evitar sobreescribir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace(".png", "")
        filename_with_timestamp = f"{base_name}_{timestamp}.png"
        
        # Crear ruta completa
        full_path = self.base_plots_dir / subfolder / filename_with_timestamp
        
        # Guardar el plot
        fig.savefig(full_path, bbox_inches='tight', dpi=300)
        
        # Registrar en el registro
        plot_info = {
            "timestamp": timestamp,
            "subfolder": subfolder,
            "original_filename": filename,
            "saved_filename": filename_with_timestamp,
            "full_path": str(full_path),
            "metadata": metadata or {}
        }
        
        plot_key = f"{subfolder}_{base_name}_{timestamp}"
        self.plot_registry[plot_key] = plot_info
        self._save_plot_registry()
        
        print(f"📊 Plot guardado: {full_path}")
        return str(full_path)
    
    def get_plots_by_category(self, subfolder):
        """Obtiene la lista de plots de una categoría específica"""
        return [
            info for info in self.plot_registry.values() 
            if info.get('subfolder') == subfolder
        ]
    
    def get_recent_plots(self, limit=10):
        """Obtiene los plots más recientes"""
        sorted_plots = sorted(
            self.plot_registry.items(),
            key=lambda x: x[1].get('timestamp', ''),
            reverse=True
        )
        return sorted_plots[:limit]
    
    def clean_old_plots(self, days_old=7):
        """Limpia plots más antiguos que el número de días especificado"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_str = cutoff_date.strftime("%Y%m%d_%H%M%S")
        
        plots_to_remove = []
        for plot_key, plot_info in self.plot_registry.items():
            if plot_info.get('timestamp', '') < cutoff_str:
                # Eliminar archivo físico
                plot_path = Path(plot_info.get('full_path', ''))
                if plot_path.exists():
                    plot_path.unlink()
                plots_to_remove.append(plot_key)
        
        # Remover del registro
        for plot_key in plots_to_remove:
            del self.plot_registry[plot_key]
        
        self._save_plot_registry()
        return len(plots_to_remove)
    
    def generate_summary_report(self):
        """Genera un reporte resumen de todos los plots"""
        summary = {
            "total_plots": len(self.plot_registry),
            "plots_by_category": {},
            "recent_activity": self.get_recent_plots(5)
        }
        
        for subfolder in self.subfolders.keys():
            category_plots = self.get_plots_by_category(subfolder)
            summary["plots_by_category"][subfolder] = len(category_plots)
        
        return summary

# Instancia global del gestor de plots
plot_manager = PlotManager()
