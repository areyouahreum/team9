import pandas as pd

def load_dataset(filepath='data/ingredients.json'):
    df = pd.read_json(filepath)
    ingredient_cols = df.columns[1:-1]
    return df, ingredient_cols