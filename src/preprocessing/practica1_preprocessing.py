import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from feature_engine.encoding import CountFrequencyEncoder


class Practica1Preprocess:
    """
    Clase de preprocesamiento alternativa a BasePreprocess.

    Cambios principales respecto a la clase base:
    - Usa variables_withExperts.xlsx
    - Imputación distinta:
        * numéricas -> mediana
        * categóricas -> most_frequent
    - Categóricas:
        * grade y sub_grade con OrdinalEncoder
        * resto con CountFrequencyEncoder
    - Numéricas:
        * RobustScaler
    - Nuevas features:
        * fico_mean
        * installment_income_ratio
        * loan_income_ratio
        * earliest_cr_line_year
        * earliest_cr_line_month
    """

    def __init__(self, var_to_process, target):
        # Leemos el Excel de variables y nos quedamos solo con las posibles predictoras
        self.raw_predictors_vars = pd.read_excel(var_to_process)
        self.raw_predictors_vars = (
            self.raw_predictors_vars
            .query("posible_predictora == 'si'")
            .variable
            .tolist()
        )

        self.target_var = target

        # Variables especiales que queremos tratar como ordinales
        self.grade_var = "grade"
        self.sub_grade_var = "sub_grade"

        # Objetos que aprenderemos en fit()
        self.num_imputer = None
        self.cat_imputer = None
        self.ordinal_encoder = None
        self.freq_encoder = None
        self.scaler = None

        # Listas de variables que guardaremos en fit()
        self.predictor_vars_used = None
        self.numeric_vars = None
        self.categoric_vars = None
        self.ordinal_vars = None
        self.freq_vars = None
        self.final_numeric_vars = None

    def _create_features(self, X):
        """
        Crea nuevas variables a partir de columnas ya existentes.
        Esta función se usa tanto en fit() como en transform().
        """
        X = X.copy()

        # Feature 1: FICO medio
        if "fico_range_low" in X.columns and "fico_range_high" in X.columns:
            X["fico_mean"] = (X["fico_range_low"] + X["fico_range_high"]) / 2

        # Feature 2: cuota / ingreso mensual
        if "installment" in X.columns and "annual_inc" in X.columns:
            ingreso_mensual = X["annual_inc"] / 12
            ingreso_mensual = ingreso_mensual.replace(0, np.nan)
            X["installment_income_ratio"] = X["installment"] / ingreso_mensual

        # Feature 3: préstamo / ingreso anual
        if "loan_amnt" in X.columns and "annual_inc" in X.columns:
            annual_inc_nonzero = X["annual_inc"].replace(0, np.nan)
            X["loan_income_ratio"] = X["loan_amnt"] / annual_inc_nonzero

        # Feature 4 y 5: año y mes de earliest_cr_line
        if "earliest_cr_line" in X.columns:
            fecha = pd.to_datetime(
                X["earliest_cr_line"],
                format="%b-%Y",
                errors="coerce"
            )
            X["earliest_cr_line_year"] = fecha.dt.year
            X["earliest_cr_line_month"] = fecha.dt.month

        return X

    def fit(self, data):
        """
        Aprende todos los parámetros del preprocesamiento usando SOLO train.
        """
        # Leemos el csv de entrenamiento
        df = pd.read_csv(data)

        # Nos quedamos solo con las predictoras que realmente existan en el csv
        self.predictor_vars_used = [
            col for col in self.raw_predictors_vars if col in df.columns
        ]

        # Separamos X e y
        X_train = df[self.predictor_vars_used].copy()
        y_train = (df[self.target_var] != "Fully Paid").astype(int)

        # Creamos nuevas features
        X_train = self._create_features(X_train)

        # Detectamos variables numéricas y categóricas
        self.numeric_vars = X_train.select_dtypes(include=["number"]).columns.tolist()
        self.categoric_vars = X_train.select_dtypes(include=["object"]).columns.tolist()

        # Variables ordinales solo si existen
        self.ordinal_vars = [
            col for col in [self.grade_var, self.sub_grade_var] if col in X_train.columns
        ]

        # Variables categóricas restantes para frequency encoding
        self.freq_vars = [
            col for col in self.categoric_vars if col not in self.ordinal_vars
        ]

        # Imputadores
        if len(self.numeric_vars) > 0:
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(X_train[self.numeric_vars])

        if len(self.categoric_vars) > 0:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(X_train[self.categoric_vars])

        # Imputamos una copia para poder ajustar encoders y scaler
        X_fit = X_train.copy()

        if len(self.numeric_vars) > 0:
            X_fit[self.numeric_vars] = self.num_imputer.transform(X_fit[self.numeric_vars])

        if len(self.categoric_vars) > 0:
            X_fit[self.categoric_vars] = self.cat_imputer.transform(X_fit[self.categoric_vars])

        # OrdinalEncoder para grade y sub_grade
        if len(self.ordinal_vars) > 0:
            categories_map = {
                "grade": ["A", "B", "C", "D", "E", "F", "G"],
                "sub_grade": [
                    "A1", "A2", "A3", "A4", "A5",
                    "B1", "B2", "B3", "B4", "B5",
                    "C1", "C2", "C3", "C4", "C5",
                    "D1", "D2", "D3", "D4", "D5",
                    "E1", "E2", "E3", "E4", "E5",
                    "F1", "F2", "F3", "F4", "F5",
                    "G1", "G2", "G3", "G4", "G5"
                ]
            }

            categories = [categories_map[var] for var in self.ordinal_vars]

            self.ordinal_encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            self.ordinal_encoder.fit(X_fit[self.ordinal_vars])

        # Frequency encoding para el resto de categóricas
        if len(self.freq_vars) > 0:
            self.freq_encoder = CountFrequencyEncoder(
                encoding_method="frequency",
                variables=self.freq_vars
            )
            self.freq_encoder.fit(X_fit[self.freq_vars], y_train)

        # Aplicamos encoders sobre una copia para saber qué variables numéricas finales escalar
        X_encoded = X_fit.copy()

        if len(self.ordinal_vars) > 0:
            X_encoded[self.ordinal_vars] = self.ordinal_encoder.transform(
                X_encoded[self.ordinal_vars]
            )

        if len(self.freq_vars) > 0:
            X_freq = self.freq_encoder.transform(X_encoded[self.freq_vars])
            X_freq = X_freq.fillna(0)
            X_encoded[self.freq_vars] = X_freq[self.freq_vars]

        # Por seguridad, sustituimos infinitos por NaN y rellenamos NaN residuales
        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
        X_encoded = X_encoded.fillna(0)

        # Recalculamos numéricas finales tras los encodings
        self.final_numeric_vars = X_encoded.select_dtypes(include=["number"]).columns.tolist()

        # Escalado numérico con RobustScaler
        if len(self.final_numeric_vars) > 0:
            self.scaler = RobustScaler()
            self.scaler.fit(X_encoded[self.final_numeric_vars])

    def transform(self, data):
        """
        Aplica el preprocesamiento usando solo lo aprendido en fit().
        """
        # Leemos el csv
        df = pd.read_csv(data)

        # Separamos X e y
        X_data = df[self.predictor_vars_used].copy()
        y_data = (df[self.target_var] != "Fully Paid").astype(int)

        # Creamos nuevas features
        X_data = self._create_features(X_data)

        # Imputación
        if len(self.numeric_vars) > 0:
            X_data[self.numeric_vars] = self.num_imputer.transform(X_data[self.numeric_vars])

        if len(self.categoric_vars) > 0:
            X_data[self.categoric_vars] = self.cat_imputer.transform(X_data[self.categoric_vars])

        # Encoding ordinal
        if len(self.ordinal_vars) > 0:
            X_data[self.ordinal_vars] = self.ordinal_encoder.transform(
                X_data[self.ordinal_vars]
            )

        # Frequency encoding
        if len(self.freq_vars) > 0:
            X_freq = self.freq_encoder.transform(X_data[self.freq_vars])
            X_freq = X_freq.fillna(0)
            X_data[self.freq_vars] = X_freq[self.freq_vars]

        # Escalado final
        if len(self.final_numeric_vars) > 0:
            X_data[self.final_numeric_vars] = self.scaler.transform(
                X_data[self.final_numeric_vars]
            )

        # Por seguridad, sustituimos infinitos por NaN y luego rellenamos cualquier NaN residual
        X_data = X_data.replace([np.inf, -np.inf], np.nan)
        X_data = X_data.fillna(0)

        return X_data, y_data

    def print_summary(self):
        print("=" * 60)
        print("RESUMEN DEL PREPROCESAMIENTO PRACTICA 1")
        print("=" * 60)
        print(f"Variables predictoras usadas: {len(self.predictor_vars_used)}")
        print(f"Variables numéricas originales: {len(self.numeric_vars)}")
        print(f"Variables categóricas originales: {len(self.categoric_vars)}")
        print(f"Variables ordinales: {self.ordinal_vars}")
        print(f"Variables con frequency encoding: {len(self.freq_vars)}")
        print(f"Variables numéricas finales escaladas: {len(self.final_numeric_vars)}")
        print("=" * 60)