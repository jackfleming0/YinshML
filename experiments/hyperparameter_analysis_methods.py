"""Additional methods for hyperparameter analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between hyperparameters."""
    original_features = df.columns.tolist()
    
    # Add pairwise interactions for numerical features
    for i, col1 in enumerate(original_features):
        for col2 in original_features[i+1:]:
            if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
    
    return df


def correlation_analysis(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Perform correlation analysis between hyperparameters and target."""
    results = {}
    
    for col in X.columns:
        try:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(X[col], y)
            
            # Spearman correlation (rank-based, handles non-linear relationships)
            spearman_corr, spearman_p = spearmanr(X[col], y)
            
            # Use the correlation with higher absolute value
            if abs(pearson_corr) > abs(spearman_corr):
                correlation = pearson_corr
                p_value = pearson_p
            else:
                correlation = spearman_corr
                p_value = spearman_p
            
            results[col] = {
                'correlation': correlation,
                'p_value': p_value,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr
            }
            
        except Exception as e:
            print(f"Correlation analysis failed for {col}: {e}")
            results[col] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'pearson_corr': 0.0,
                'spearman_corr': 0.0
            }
    
    return results


def random_forest_analysis(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """Perform Random Forest feature importance analysis."""
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X_scaled, y)
    
    # Get feature importances
    feature_importances = rf.feature_importances_
    
    # Calculate model performance
    y_pred = rf.predict(X_scaled)
    r2_score_val = r2_score(y, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_scaled, y, cv=min(5, len(X)), scoring='r2')
    
    return {
        'feature_importances': dict(zip(X.columns, feature_importances)),
        'r2_score': r2_score_val,
        'cv_score_mean': np.mean(cv_scores),
        'cv_score_std': np.std(cv_scores),
        'model': rf,
        'scaler': scaler
    }


def lasso_analysis(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """Perform LASSO regression analysis for feature selection."""
    # Scale features (important for LASSO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use cross-validation to find best alpha
    lasso_cv = LassoCV(cv=min(5, len(X)), random_state=random_state, max_iter=1000)
    lasso_cv.fit(X_scaled, y)
    
    # Train final model with best alpha
    lasso = Lasso(alpha=lasso_cv.alpha_, random_state=random_state, max_iter=1000)
    lasso.fit(X_scaled, y)
    
    # Get coefficients (absolute values as importance)
    coefficients = np.abs(lasso.coef_)
    
    # Calculate model performance
    y_pred = lasso.predict(X_scaled)
    r2_score_val = r2_score(y, y_pred)
    
    return {
        'feature_importances': dict(zip(X.columns, coefficients)),
        'r2_score': r2_score_val,
        'alpha': lasso_cv.alpha_,
        'n_features_selected': np.sum(coefficients > 0),
        'model': lasso,
        'scaler': scaler
    }


def create_correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Create correlation matrix for all hyperparameters."""
    corr_matrix = df.corr()
    
    # Convert to nested dictionary
    result = {}
    for i, row_name in enumerate(corr_matrix.index):
        result[row_name] = {}
        for j, col_name in enumerate(corr_matrix.columns):
            result[row_name][col_name] = float(corr_matrix.iloc[i, j])
    
    return result 