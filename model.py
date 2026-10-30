"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    """
        Đây là bước kiểm tra chuyển đổi các giá trị Nan ở trong ma trận đầu vào X
        Nếu một cột (feature) toàn giá trị Nan thì chuyển hết thành 0.0
        Nếu một cột(feature) lẫn giá trị Nan với giá trị số thì gán giá trị Nan bằng trung bình các giá trị số trong cột
    """
    X = np.asarray(X)
    all_nan_col = np.all(np.isnan(X), axis = 0)
    all_nan_col_idx = np.where(all_nan_col)[0]
    N, F = X.shape
    for j in range(F):
        if j in all_nan_col_idx:
            X[:, j] = np.nan_to_num(X[:, j], nan = 0.0)
        else:
            X[:, j] = np.nan_to_num(X[:, j], nan = np.nanmean(X[:, j]))
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    """
        Phương pháp thống kê nhằm xác định giá trị ngoại lai 
        Phương pháp này xác định giá trị ngoại lai bằng cách tính 
            Tứ phân vị thứ nhất Q1: tức là 25% giá trị nhỏ hơn hoặc bằng giá trị này
            Tứ phân vị thứ ba Q3: tức là 50% giá trị nhỏ hơn hoặc bằng giá trị này
        Sau đó xây dựng 2 khoảng
            Lower Bound = Q1 - k * (Q3 - Q1)
            Upper Bound = Q3 + k * (Q3 - Q1) 
    """
    Q1 = np.percentile(X, 25, axis = 0)
    Q3 = np.percentile(X, 75, axis = 0)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR 
    upper_bound = Q3 + k * IQR 
    return lower_bound, upper_bound

# Step 3 - clip_columns
import numpy as np
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    """
    Sau khi đã có 2 giá trị giới hạn lower upper thì sẽ cắt trực tiếp lên ma trận đầu vào X
    """
    m, n = len(X), len(X[0])
    X = np.asarray(X)
    for i in range(m):
        for j in range(n):
            if X[i, j] < lower[j]:
                X[i, j] = lower[j]
            if X[i, j] > upper[j]:
                X[i, j] = upper[j]
    return X.tolist()

# Step 4 - make_ratio_feature (not yet solved)
# TODO: implement

# Step 5 - append_column (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

