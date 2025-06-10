import pandas as pd

def load_and_clean_excel(file_path):
    df_raw = pd.read_excel(file_path, sheet_name='Sheet1')
    df_clean = df_raw.drop(index=[0, 1]).reset_index(drop=True)
    df_clean.columns = df_raw.iloc[1]
    df_clean = df_clean.drop(index=0).reset_index(drop=True)
    return df_clean