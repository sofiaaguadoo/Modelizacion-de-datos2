import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from functools import partial

class Practica1Filtering:
    """
    Clase de filtrado alternativa a BaseFiltering.

    Pasos:
    1. Eliminar variables de varianza muy baja
    2. Seleccionar las k mejores variables según mutual information
    """

    def __init__(self, variance_threshold=0.0, k_best=100):
        self.variance_threshold_value = variance_threshold
        self.k_best = k_best

        self.var_filter = VarianceThreshold(threshold=self.variance_threshold_value)
        self.kbest_filter = None

        self.features_after_variance = None
        self.selected_features = None

        self.n_features_initial = None
        self.n_features_after_variance = None
        self.n_features_final = None

    def fit(self, X_data, y_data):
        """
        Ajusta los filtros usando SOLO train.
        """
        self.n_features_initial = X_data.shape[1]

        # Paso 1: filtro por varianza
        X_var = self.var_filter.fit_transform(X_data)
        self.features_after_variance = X_data.columns[self.var_filter.get_support()].tolist()
        self.n_features_after_variance = len(self.features_after_variance)

        X_var_df = pd.DataFrame(X_var, columns=self.features_after_variance, index=X_data.index)

        # Ajustamos k para no pedir más variables de las que existen
        k_real = min(self.k_best, X_var_df.shape[1])

        # Paso 2: SelectKBest con información mutua
        self.kbest_filter = SelectKBest(score_func=partial(mutual_info_classif, random_state=42),k=k_real)
        self.kbest_filter.fit(X_var_df, y_data.values.ravel())

        self.selected_features = X_var_df.columns[self.kbest_filter.get_support()].tolist()
        self.n_features_final = len(self.selected_features)

    def transform(self, X_data):
        """
        Aplica los filtros usando solo lo aprendido en fit().
        """
        X_var = self.var_filter.transform(X_data)
        X_var_df = pd.DataFrame(X_var, columns=self.features_after_variance, index=X_data.index)

        X_selected = self.kbest_filter.transform(X_var_df)
        X_selected_df = pd.DataFrame(X_selected, columns=self.selected_features, index=X_data.index)

        return X_selected_df

    def print_summary(self):
        print("=" * 60)
        print("RESUMEN DEL FILTRADO PRACTICA 1")
        print("=" * 60)
        print(f"Features iniciales:                    {self.n_features_initial}")
        print(f"Tras filtro de varianza:              {self.n_features_after_variance}")
        print(f"Features seleccionadas finales:       {self.n_features_final}")
        print("=" * 60)