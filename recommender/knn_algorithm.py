from sklearn.neighbors import NearestNeighbors
import numpy as np

def get_user_vector(ingredient_cols, user_input_list):
    return [1 if ing in user_input_list else 0 for ing in ingredient_cols]

def knn_algorithm(df, X, user_vector, n_neighbors=3):
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(X)
    d, i = knn.kneighbors([user_vector])
    results = df.iloc[i[0]].copy()
    results['match'] = (1 - d[0])
    return results[['recipe', 'match']]

# For strict match (all recipe ingredients must be in user's list)
def strict_matches(df, ingredient_cols, user_ings):
    def is_strict_match(row):
        recipe_ings = [ing for ing, val in row.items() if val == 1]
        return all(ing in user_ings for ing in recipe_ings)

    return df[df[ingredient_cols].apply(is_strict_match, axis=1)]
