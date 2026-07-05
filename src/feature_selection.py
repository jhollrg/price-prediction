"""SHAP-based feature selector with a scikit-learn-compatible API.

Uses CatBoost's native SHAP computation (get_feature_importance type='ShapValues')
which runs in CatBoost's C++ core and avoids version-skew issues between the
catboost and shap packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ShapFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features by mean absolute SHAP value, backed by CatBoostRegressor.

    Fits CatBoost internally, computes SHAP values via CatBoost's native
    ``get_feature_importance(type='ShapValues')``, and keeps only features
    whose mean |SHAP| meets the selection criterion.

    Implements the full scikit-learn ``TransformerMixin`` interface so it can
    be dropped into a ``Pipeline``.  ``fit_transform`` is inherited.

    Parameters
    ----------
    threshold : float
        Minimum mean |SHAP| a feature must have to survive selection.
        Ignored when ``n_features_to_select`` is set.
    n_features_to_select : int, optional
        If given, keep exactly the top-k features by mean |SHAP|, regardless
        of their absolute magnitude.  Mutually exclusive with ``threshold``.
    cat_features : list of str, optional
        Column names to treat as categoricals in the CatBoost Pool.
    catboost_params : dict, optional
        Key-value pairs merged on top of the default CatBoostRegressor
        configuration.  Use this to override iterations, depth, etc.
    shap_plot_max_rows : int
        Maximum number of rows stored for the beeswarm plot.  When the
        training set is larger, a random subsample of this size is used.
        Does not affect importance scores, which are computed on all rows.
    random_state : int
        Seed for CatBoost and for the plot-subsample RNG.

    Attributes
    ----------
    estimator_ : CatBoostRegressor
        Fitted CatBoost model.
    feature_names_in_ : list[str]
        Column names seen during ``fit``.
    feature_importances_ : np.ndarray, shape (n_features,)
        Mean absolute SHAP value per feature, in original column order.
    selected_features_ : list[str]
        Names of features that passed the selection criterion.
    shap_values_ : np.ndarray, shape (n_plot_rows, n_features)
        SHAP matrix for the plot subsample (or all rows if small enough).
    X_train_ : pd.DataFrame
        Feature rows corresponding to ``shap_values_`` — kept for beeswarm.
    expected_value_ : float
        Model base value (mean prediction), i.e. the SHAP bias term.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        n_features_to_select: Optional[int] = None,
        cat_features: Optional[list[str]] = None,
        catboost_params: Optional[dict[str, Any]] = None,
        shap_plot_max_rows: int = 2_000,
        random_state: int = 42,
    ) -> None:
        self.threshold = threshold
        self.n_features_to_select = n_features_to_select
        self.cat_features = cat_features
        self.catboost_params = catboost_params
        self.shap_plot_max_rows = shap_plot_max_rows
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "ShapFeatureSelector":
        """Fit CatBoost, compute SHAP values, and determine selected features.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series
            Target values.  Required (not optional despite the sklearn
            signature convention).

        Returns
        -------
        self
            Fitted selector with ``feature_importances_`` and
            ``selected_features_`` populated.

        Raises
        ------
        ValueError
            If ``y`` is ``None``.
        """
        from catboost import CatBoostRegressor, Pool

        if y is None:
            raise ValueError("y is required; ShapFeatureSelector is supervised.")

        self.feature_names_in_: list[str] = list(X.columns)

        cat_cols = [c for c in (self.cat_features or []) if c in X.columns]
        cat_col_idx = [self.feature_names_in_.index(c) for c in cat_cols]

        pool = Pool(
            data=X,
            label=y,
            cat_features=cat_col_idx or None,
            feature_names=self.feature_names_in_,
        )

        cb_params: dict[str, Any] = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": self.random_state,
            "verbose": 0,
            # Prevent CatBoost from writing training artefacts to disk
            "allow_writing_files": False,
        }
        if self.catboost_params:
            cb_params.update(self.catboost_params)

        self.estimator_ = CatBoostRegressor(**cb_params)
        self.estimator_.fit(pool)

        # Shape: (n_samples, n_features + 1); final column is the bias term
        # (expected prediction), not a feature SHAP value.
        shap_matrix: np.ndarray = self.estimator_.get_feature_importance(
            pool, type="ShapValues"
        )
        self.expected_value_: float = float(shap_matrix[:, -1].mean())
        raw_shap: np.ndarray = shap_matrix[:, :-1]

        self.feature_importances_: np.ndarray = np.abs(raw_shap).mean(axis=0)

        if len(X) > self.shap_plot_max_rows:
            rng = np.random.default_rng(self.random_state)
            plot_idx = rng.choice(len(X), size=self.shap_plot_max_rows, replace=False)
            self.shap_values_: np.ndarray = raw_shap[plot_idx]
            self.X_train_: pd.DataFrame = X.iloc[plot_idx].reset_index(drop=True)
        else:
            self.shap_values_ = raw_shap
            self.X_train_ = X.copy()

        self._apply_selection()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *X* containing only the selected features.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain every column in ``selected_features_``.

        Returns
        -------
        pd.DataFrame
            Column-subset of *X*, in selection order.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        ValueError
            If any selected column is absent from *X*.
        """
        check_is_fitted(self, "selected_features_")

        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"X is missing columns required by the selector: {missing}"
            )

        return X[self.selected_features_].copy()

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.01,
    ) -> "ShapFeatureSelector":
        """Fit and select in one call, overriding the instance threshold.

        Sets ``self.threshold = threshold`` and ``self.n_features_to_select =
        None`` so the threshold criterion takes precedence, then calls
        ``fit(X, y)``.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : pd.Series
            Target values.
        threshold : float
            Minimum mean |SHAP| required for a feature to survive.

        Returns
        -------
        self
        """
        self.threshold = threshold
        self.n_features_to_select = None
        return self.fit(X, y)

    def save_shap_plot(
        self,
        output_path: Union[str, Path],
        top_n: int = 20,
    ) -> None:
        """Render the SHAP beeswarm plot and save it to disk.

        Each point is one sample.  Horizontal position encodes the SHAP value
        (impact on prediction); colour encodes the raw feature value (red =
        high, blue = low).  Features are sorted by mean |SHAP| descending.

        Parameters
        ----------
        output_path : str or Path
            Destination file.  Parent directories are created if needed.
            The extension determines the format (use ``.png`` for README use).
        top_n : int
            Maximum number of features shown on the y-axis.  Capped at the
            number of features seen during ``fit``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        import shap

        check_is_fitted(self, "shap_values_")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        top_n = min(top_n, len(self.feature_names_in_))

        # Underscores → spaces for legible axis labels
        display_names = [n.replace("_", " ") for n in self.feature_names_in_]

        # shap.summary_plot creates and sizes its own figure when show=False;
        # it leaves plt.gcf() pointing at that figure afterwards.
        shap.summary_plot(
            self.shap_values_,
            self.X_train_,
            feature_names=display_names,
            max_display=top_n,
            show=False,
            sort=True,
            color_bar_label="Feature value",
        )

        fig = plt.gcf()

        # axes[0] is the beeswarm scatter; axes[1] (when present) is the
        # colorbar — set the title on the scatter axis explicitly.
        fig.axes[0].set_title(
            "Feature importance (SHAP values) — NYC Taxi Fare",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_importance(
        self,
        top_n: int = 20,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Render a mean-|SHAP| horizontal bar chart.

        Simpler than the beeswarm: one bar per feature, length = mean |SHAP|.
        Useful for quick comparisons without needing per-sample scatter data.

        Parameters
        ----------
        top_n : int
            Number of top features to display.
        save_path : str or Path, optional
            If given, save to this path at 150 DPI and close the figure.
            If ``None``, call ``plt.show()`` instead.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before ``fit``.
        """
        check_is_fitted(self, "feature_importances_")

        top_n = min(top_n, len(self.feature_names_in_))

        ranked_idx = np.argsort(self.feature_importances_)[::-1][:top_n]
        names  = [self.feature_names_in_[i].replace("_", " ") for i in ranked_idx]
        values = self.feature_importances_[ranked_idx]

        fig_height = max(4, top_n * 0.38 + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_height))

        y_pos = np.arange(top_n)
        ax.barh(y_pos, values[::-1], align="center", color="#1f77b4", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(
            "Feature importance (SHAP values) — NYC Taxi Fare",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _apply_selection(self) -> None:
        """Populate ``selected_features_`` from importances + current criterion."""
        if self.n_features_to_select is not None:
            k = min(self.n_features_to_select, len(self.feature_names_in_))
            top_idx = np.argsort(self.feature_importances_)[::-1][:k]
            # Preserve original column order so downstream pipelines are stable
            self.selected_features_: list[str] = [
                self.feature_names_in_[i] for i in sorted(top_idx)
            ]
        else:
            thr = self.threshold if self.threshold is not None else 0.01
            self.selected_features_ = [
                name
                for name, imp in zip(
                    self.feature_names_in_, self.feature_importances_
                )
                if imp >= thr
            ]
