import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_shap_plot(fig, filename, subfolder="shap_explanations"):
    """Guarda automáticamente los plots SHAP en la carpeta plots/"""
    plots_dir = os.path.join("plots", subfolder)
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = filename.replace(".png", "")
    full_filename = f"{base_name}_{timestamp}.png"
    
    filepath = os.path.join(plots_dir, full_filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    return filepath

def explain_prediction_shap(model, X_new, feature_names, save_plot_flag=True):
    """Genera explicaciones SHAP para la predicción actual"""
    xgb_model = model.named_steps['model']
    X_scaled = model.named_steps['scaler'].transform(X_new)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)
    
    plt.figure(figsize=(10, 4))
    
    if isinstance(shap_values, list):
        shap_plot = shap.force_plot(
            explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_new.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
    else:
        shap_plot = shap.force_plot(
            explainer.expected_value,
            shap_values,
            X_new.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
    
    fig = plt.gcf()
    
    # Guardar automáticamente el plot SHAP
    if save_plot_flag:
        save_shap_plot(fig, "shap_force_plot.png")
    
    return fig
