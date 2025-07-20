import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plot(model, df_encoded, selected_index, class_index, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_encoded)

    fig, ax = plt.subplots()

    if isinstance(shap_values, list):
        values = shap_values[class_index][selected_index]
    else:
        values = shap_values[selected_index]

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[class_index]
    else:
        base_value = explainer.expected_value

    shap.plots.waterfall(shap.Explanation(
        values=values,
        base_values=base_value,
        data=df_encoded.iloc[selected_index],
        feature_names=feature_names
    ), show=False)
    return fig, shap_values, explainer