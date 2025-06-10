import shap
import matplotlib.pyplot as plt

def generate_shap_plot(model, df_encoded, selected_index, class_index, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_encoded)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[class_index][selected_index],
        base_values=explainer.expected_value[class_index],
        data=df_encoded.iloc[selected_index],
        feature_names=feature_names
    ), show=False)
    return fig, shap_values, explainer