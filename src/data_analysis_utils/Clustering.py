
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import hdbscan

import joblib
import pandas as pd
import numpy as np



class ClusterFeatureModels:
    """
    Fit/predict hard cluster labels for KMeans, GaussianMixture, and HDBSCAN.

    Input contract behavior:
    - If fit() receives a DataFrame, the fitted feature column names are saved.
    - predict() then requires those trained columns to exist and reorders input
      columns to match training order.
    - Extra columns are allowed by default.
    - If fit() used non-DataFrame input, only feature-count validation is applied.
    """

    def __init__(self, n_kmeans=None, 
                 n_gmm=None, 
                 random_state=42):
        self.n_kmeans = n_kmeans
        self.n_gmm = n_gmm
        self.random_state = random_state

        self.kmeans = None
        self.gmm = None
        self.hdb = None

        self.hdb_noise_label_ = None
        self.hdb_params_ = None
        self.selection_results_ = {}

        self.feature_names_ = [
            "cluster_kmeans",
            "cluster_gmm",
            "cluster_hdbscan",
        ]

        # Input contract metadata (persisted via joblib)
        self.expected_input_columns_ = None
        self.n_input_features_ = None

        # Policy: allow extras, require DataFrame when named contract exists
        self.allow_extra_columns = True
        self.require_dataframe_for_named_contract = True

    @staticmethod
    def _to_dense(X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    @staticmethod
    def _safe_silhouette(X, labels):
        unique = np.unique(labels)
        if unique.shape[0] < 2:
            return np.nan
        return float(silhouette_score(X, labels))

    def _capture_input_contract(self, X):
        if hasattr(X, "columns"):
            expected_cols = list(X.columns)
            self.expected_input_columns_ = expected_cols
            self.n_input_features_ = len(expected_cols)
            return

        Xd = self._to_dense(X)
        if Xd.ndim != 2:
            raise ValueError("Input data must be 2D")
        self.expected_input_columns_ = None
        self.n_input_features_ = int(Xd.shape[1])

    def _validate_and_align_input(self, X, stage="predict"):
        expected_cols = getattr(self, "expected_input_columns_", None)
        n_input_features = getattr(self, "n_input_features_", None)
        allow_extra = getattr(self, "allow_extra_columns", True)
        require_df_named = getattr(self, "require_dataframe_for_named_contract", True)

        if expected_cols is not None:
            if not hasattr(X, "columns"):
                if require_df_named:
                    raise TypeError(
                        f"{stage} expects a pandas DataFrame with trained columns in order-aware mode"
                    )
                Xd = self._to_dense(X)
            else:
                missing = [col for col in expected_cols if col not in X.columns]
                if missing:
                    raise ValueError(f"Missing required columns at {stage} time: {missing}")
                if not allow_extra:
                    extra = [col for col in X.columns if col not in expected_cols]
                    if extra:
                        raise ValueError(f"Unexpected extra columns at {stage} time: {extra}")
                Xd = self._to_dense(X.loc[:, expected_cols])
        else:
            Xd = self._to_dense(X)

        if Xd.ndim != 2:
            raise ValueError(f"Input data at {stage} time must be 2D")

        if n_input_features is not None and Xd.shape[1] != int(n_input_features):
            raise ValueError(
                f"Expected {int(n_input_features)} input features at {stage} time, got {Xd.shape[1]}"
            )

        return Xd

    def choose_kmeans_k(self, X_train, k_values=range(2, 13)):
        Xd = self._validate_and_align_input(X_train, stage="choose_kmeans_k")
        rows = []

        for k in k_values:
            km = KMeans(
                n_clusters=int(k),
                random_state=self.random_state,
                n_init="auto",
            )
            labels = km.fit_predict(Xd)
            rows.append(
                {
                    "k": int(k),
                    "inertia": float(km.inertia_),
                    "silhouette": self._safe_silhouette(Xd, labels),
                }
            )

        df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        best_row = df.sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]
        best_k = int(best_row["k"])

        self.selection_results_["kmeans"] = df
        self.n_kmeans = best_k
        return best_k, df

    def choose_gmm_components(self, X_train, n_values=range(2, 13), covariance_types=("full", "diag")):
        Xd = self._validate_and_align_input(X_train, stage="choose_gmm_components")
        rows = []

        for n in n_values:
            for cov in covariance_types:
                gm = GaussianMixture(
                    n_components=int(n),
                    covariance_type=cov,
                    random_state=self.random_state,
                )
                gm.fit(Xd)
                rows.append(
                    {
                        "n_components": int(n),
                        "covariance_type": cov,
                        "bic": float(gm.bic(Xd)),
                        "aic": float(gm.aic(Xd)),
                    }
                )

        df = pd.DataFrame(rows).sort_values(["bic", "aic"], ascending=[True, True]).reset_index(drop=True)
        best = df.iloc[0]

        self.selection_results_["gmm"] = df
        self.n_gmm = int(best["n_components"])
        return int(best["n_components"]), str(best["covariance_type"]), df

    def choose_hdbscan_params(
        self,
        X_train,
        min_cluster_size_values=(20, 40, 60),
        min_samples_values=(5, 10, 15),
    ):
        Xd = self._validate_and_align_input(X_train, stage="choose_hdbscan_params")
        rows = []

        for min_cluster_size in min_cluster_size_values:
            for min_samples in min_samples_values:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    prediction_data=True,
                )
                model.fit(Xd)
                labels = model.labels_

                noise_ratio = float(np.mean(labels == -1))
                non_noise_mask = labels != -1
                non_noise_labels = labels[non_noise_mask]

                n_clusters = int(np.unique(non_noise_labels).shape[0])
                if non_noise_mask.sum() > 2 and n_clusters > 1:
                    sil = self._safe_silhouette(Xd[non_noise_mask], non_noise_labels)
                else:
                    sil = np.nan

                rows.append(
                    {
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "n_clusters": n_clusters,
                        "noise_ratio": noise_ratio,
                        "silhouette_non_noise": sil,
                    }
                )

        df = pd.DataFrame(rows)
        df["silhouette_non_noise"] = df["silhouette_non_noise"].fillna(-1)
        df = df.sort_values(["silhouette_non_noise", "noise_ratio", "n_clusters"], ascending=[False, True, False]).reset_index(drop=True)

        best = df.iloc[0]
        best_params = {
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
        }

        self.selection_results_["hdbscan"] = df
        self.hdb_params_ = best_params
        return best_params, df

    def fit(
        self,
        X_train,
        auto_select=True,
        k_values=range(2, 13),
        gmm_n_values=range(2, 13),
        gmm_covariance_types=("full", "diag"),
        hdbscan_min_cluster_size_values=(20, 40, 60),
        hdbscan_min_samples_values=(5, 10, 15),
    ):
        self._capture_input_contract(X_train)
        Xd = self._validate_and_align_input(X_train, stage="fit")

        if auto_select:
            if self.n_kmeans is None:
                self.choose_kmeans_k(X_train, k_values=k_values)
            if self.n_gmm is None:
                best_gmm_n, best_cov, _ = self.choose_gmm_components(
                    X_train,
                    n_values=gmm_n_values,
                    covariance_types=gmm_covariance_types,
                )
            else:
                best_cov = "full"

            if self.hdb_params_ is None:
                self.choose_hdbscan_params(
                    X_train,
                    min_cluster_size_values=hdbscan_min_cluster_size_values,
                    min_samples_values=hdbscan_min_samples_values,
                )
        else:
            best_cov = "full"
            if self.n_kmeans is None:
                raise ValueError("Set n_kmeans when auto_select=False")
            if self.n_gmm is None:
                raise ValueError("Set n_gmm when auto_select=False")
            if self.hdb_params_ is None:
                self.hdb_params_ = {"min_cluster_size": 40, "min_samples": 15}

        self.kmeans = KMeans(
            n_clusters=int(self.n_kmeans),
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans.fit(Xd)

        self.gmm = GaussianMixture(
            n_components=int(self.n_gmm),
            covariance_type=best_cov,
            random_state=self.random_state,
        )
        self.gmm.fit(Xd)

        self.hdb = hdbscan.HDBSCAN(
            min_cluster_size=int(self.hdb_params_["min_cluster_size"]),
            min_samples=int(self.hdb_params_["min_samples"]),
            prediction_data=True,
        )
        self.hdb.fit(Xd)
        self.hdb.generate_prediction_data()

        max_hdb_label = self.hdb.labels_[self.hdb.labels_ >= 0].max() if np.any(self.hdb.labels_ >= 0) else 0
        self.hdb_noise_label_ = int(max_hdb_label + 1)
        return self

    def predict(self, X):
        Xd = self._validate_and_align_input(X, stage="predict")

        kmeans_labels = self.kmeans.predict(Xd).astype(int)
        gmm_labels = self.gmm.predict(Xd).astype(int)

        hdb_labels, _ = hdbscan.approximate_predict(self.hdb, Xd)
        hdb_labels = hdb_labels.astype(int)
        hdb_labels[hdb_labels < 0] = self.hdb_noise_label_

        return pd.DataFrame(
            {
                "cluster_kmeans": kmeans_labels,
                "cluster_gmm": gmm_labels,
                "cluster_hdbscan": hdb_labels,
            }
        )

    def get_feature_names_out(self):
        return list(self.feature_names_)

    def get_expected_input_columns(self):
        expected_cols = getattr(self, "expected_input_columns_", None)
        return None if expected_cols is None else list(expected_cols)

    def save(self, path="cluster_feature_models.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="cluster_feature_models.joblib"):
        return joblib.load(path)
    

"""
# Run this after Xtrain/Xtest exist (and after your standard scaling step).
cluster_models = ClusterFeatureModels(
    n_kmeans=None,
    n_gmm=None,
    random_state=42,
).fit(
    Xtrain_small,
    auto_select=True,
    k_values=range(2, 13),
    gmm_n_values=range(2, 13),
    gmm_covariance_types=("full", "diag"),
    hdbscan_min_cluster_size_values=(20, 40, 60),
    hdbscan_min_samples_values=(5, 10, 15),
)

print("Selected KMeans k:", cluster_models.n_kmeans)
print("Selected GMM n_components:", cluster_models.n_gmm)
print("Selected HDBSCAN params:", cluster_models.hdb_params_)

print("\nTop KMeans candidates:")
print(cluster_models.selection_results_["kmeans"].sort_values(["silhouette", "inertia"], ascending=[False, True]).head(5))

print("\nTop GMM candidates:")
print(cluster_models.selection_results_["gmm"].head(5))

print("\nTop HDBSCAN candidates:")
print(cluster_models.selection_results_["hdbscan"].head(5))

train_cluster_features = cluster_models.predict(Xtrain)
test_cluster_features = cluster_models.predict(Xtest)

print(train_cluster_features.head())
print(test_cluster_features.head())

--------------------------------------------------

# Attach the 3 cluster columns to your model matrices.
# If Xtrain/Xtest are numpy arrays, convert first.
Xtrain_df = pd.DataFrame(cluster_models._to_dense(Xtrain)).reset_index(drop=True)
Xtest_df = pd.DataFrame(cluster_models._to_dense(Xtest)).reset_index(drop=True)

Xtrain_with_clusters = pd.concat(
    [Xtrain_df, train_cluster_features.reset_index(drop=True)], axis=1
)
Xtest_with_clusters = pd.concat(
    [Xtest_df, test_cluster_features.reset_index(drop=True)], axis=1
)

print(Xtrain_with_clusters.shape, Xtest_with_clusters.shape)
print(Xtrain_with_clusters[cluster_models.feature_names_].head())

# Predict hard cluster labels for any future transformed observations.
# IMPORTANT: X_future_transformed must use the exact same preprocessing pipeline as training.

def add_cluster_features_for_future(X_future_transformed, models=loaded_cluster_models):
    X_future_df = pd.DataFrame(models._to_dense(X_future_transformed)).reset_index(drop=True)
    future_cluster_features = models.predict(X_future_transformed)
    X_future_with_clusters = pd.concat(
        [X_future_df, future_cluster_features.reset_index(drop=True)], axis=1
    )
    return X_future_with_clusters

# Example:
# X_future_with_clusters = add_cluster_features_for_future(X_future_transformed)
# X_future_pred = your_classifier.predict(X_future_with_clusters)

"""