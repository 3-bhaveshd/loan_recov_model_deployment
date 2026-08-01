import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


class LoanRecoveryPipeline:
    """OOP Pipeline handling Feature Engineering, Preprocessing,

    Training, and Inference for ARC Loan Recovery.
    """

    LEAKAGE_FEATURES = [
        "recoveries",
        "collection_recovery_fee",
        "payment_plan_status",
        "recovery_status",
    ]

    CAT_ATTRIBS = [
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "term_months",
    ]

    def __init__(self):
        self.pipeline = None
        self.model = None
        self.num_attribs = None

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations and drops leakers/identifiers."""
        df_out = df.copy()

        # Drop data leaking features if present
        df_out = df_out.drop(
            self.LEAKAGE_FEATURES, axis=1, errors="ignore"
        )

        # Convert term to categorical string format
        if "term_months" in df_out.columns:
            df_out["term_months"] = df_out["term_months"].astype(str)

        # Merge FICO scores into a single composite feature
        if (
            "fico_range_low" in df_out.columns
            and "fico_range_high" in df_out.columns
        ):
            df_out["fico_score"] = (
                df_out["fico_range_low"] + df_out["fico_range_high"]
            )
            df_out = df_out.drop(
                ["fico_range_low", "fico_range_high"], axis=1
            )

        # Create ratio of collateral to outstanding debt
        if (
            "collateral_value" in df_out.columns
            and "outstanding_principal" in df_out.columns
        ):
            df_out["collateral_value"] = df_out["collateral_value"].fillna(0)
            # Prevent division by zero
            denom = df_out["outstanding_principal"].replace(0, np.nan)
            df_out["collat_to_outprinc_ratio"] = (
                df_out["collateral_value"] / denom
            ).fillna(0)

        # Flag missing DTI and sanitize -999 placeholder values
        if "dti" in df_out.columns:
            df_out["is_dti"] = (df_out["dti"] != -999).astype(int)
            df_out["dti"] = df_out["dti"].replace(-999, np.nan)

        # Normalize casing for categorical string columns
        if "home_ownership" in df_out.columns:
            df_out["home_ownership"] = df_out["home_ownership"].str.upper()

        # Drop ID column if present
        if "loan_id" in df_out.columns:
            df_out = df_out.drop(["loan_id"], axis=1)

        return df_out

    def build_transformer(self, X: pd.DataFrame):
        """Constructs Scikit-Learn ColumnTransformer based on input data."""
        self.num_attribs = [
            c for c in X.columns if c not in self.CAT_ATTRIBS
        ]

        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        self.pipeline = ColumnTransformer(
            [
                ("num", num_pipeline, self.num_attribs),
                ("cat", cat_pipeline, self.CAT_ATTRIBS),
            ]
        )

    def fit(self, df_raw: pd.DataFrame, target_col: str = "recovery_rate"):
        """Fits feature preprocessing and the XGBoost regressor model."""
        df_engineered = self.feature_engineering(df_raw)

        y = df_engineered[target_col]
        X = df_engineered.drop(columns=[target_col], errors="ignore")

        self.build_transformer(X)
        X_prepared = self.pipeline.fit_transform(X)

        self.model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_prepared, y)
        return self

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Transforms input data and outputs clipped [0, 1] recovery rate predictions."""
        if self.pipeline is None or self.model is None:
            raise ValueError("Pipeline and model must be trained or loaded first.")

        df_engineered = self.feature_engineering(df_raw)
        X_prepared = self.pipeline.transform(df_engineered)
        raw_preds = self.model.predict(X_prepared)

        # Strictly clip to logical boundaries [0, 1]
        return np.clip(raw_preds, 0, 1)

    def save(self, filepath: str = "model_pipeline.joblib"):
        """Saves pipeline state to disk."""
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model": self.model,
                "num_attribs": self.num_attribs,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str = "model_pipeline.joblib"):
        """Loads pipeline state from disk."""
        instance = cls()
        data = joblib.load(filepath)
        instance.pipeline = data["pipeline"]
        instance.model = data["model"]
        instance.num_attribs = data["num_attribs"]
        return instance