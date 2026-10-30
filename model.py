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

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    # Bước này giúp tính tỉ lệ giữa 2 feature với nhau để giàu đặc trưng hơn
    return numerator / (denominator + eps)

# Step 5 - append_column
import numpy
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    X = np.asarray(X) 
    return np.column_stack((X, col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    """
        Đây là bước chuyển các features dạng chữ thành mã one-hot
        - Lấy tất cả các giá trị duy nhất trong labels và sắp xếp: X(N,)
        - Tạo một vector K toàn 0 với shape (N, ) và so với X với mỗi giá trị
        trong X được gán cho K với idx giá trị đó tại K = 1
    """
    label = sorted(list(set(labels)))
    dct = {}
    for i in range(len(label)):
        tmp = np.zeros(len(label))
        tmp[i] = 1
        dct[label[i]] = tmp 
    result = []
    for i in labels:
        result.append(dct[i])
    return np.array(result)

# Step 7 - fit_standardizer
import numpy as np
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.asarray(X)
    N = X.shape[0]
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std = np.where(std ==0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
import numpy as np
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N = len(X)
    newcol = np.ones(N)
    X = np.column_stack((newcol, X))
    return X

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    # Đây là bước tạo indices đã xáo trộn để tránh mô hình học vẹt
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    return indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N = len(indices)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    train_idx = indices[0: n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = X[indices]
    y_sub = y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
import numpy as np
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    X = np.asarray(X)
    y = np.asarray(y)
    theta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    y_pred = X @ theta
    return y_pred

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    error = y_pred - y_true 
    mae = np.mean(np.abs(error))
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    N = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred 
    rmse = np.sqrt(1/N * (np.sum(error ** 2)))
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_pred - y_true) ** 2)
    SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if SS_tot == 0: return 0.0
    R_2 = 1 - SS_res / SS_tot
    return R_2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    error = y_true - y_pred
    mean = np.mean(error)
    std = np.std(error, ddof = 0)
    median_abs = np.median(np.abs(error))
    return {
        "mean": mean,
        "std": float(std),
        "median_abs": median_abs
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = np.asarray(X)
    X = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X, iqr_k)
    X = clip_columns(X, lower, upper)
    return X

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(numerator, denominator, 1e-8)
    X_num = append_column(X_num, ratio)
    if cat_labels is not None:
        result = one_hot_encode(cat_labels)
        return np.hstack([X_num, result])
    return X_num

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    indices = make_shuffled_indices(len(X), seed)
    train_idx, val_idx, test_idx = partition_indices( indices, train_ratio, val_ratio)
    X_sub, y_sub = subset_xy(X, y, indices)
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx]
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean, std = fit_standardizer(splits["X_train"])
    result = {}
    
    result["X_train"] = add_bias_column(apply_standardizer(splits["X_train"], mean, std))
    result["X_val"] = add_bias_column(apply_standardizer(splits["X_val"], mean, std))
    result["X_test"] = add_bias_column(apply_standardizer(splits["X_test"], mean, std))
    for key, value in splits.items():
        if key not in result:
            result[key] = value
            
    return result, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = root_mean_squared_error(y_true, y_pred)
    r_2 = r_squared(y_true, y_pred)
    residual_s = residual_summary(y_true, y_pred)
    return {
        "mae": MAE,
        "rmse": RMSE,
        "r2": r_2,
        "residual_summary": residual_s
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X = np.asarray(X)
    y = np.asarray(y)
    X_clean = prepare_cleaned_features(X, iqr_k)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels)
    splits = make_train_val_test(X_feat, y, train_ratio. val_ratio, seed)
    result,_, _ = standardize_and_add_bias(splits)
    theta = ols_fit(result["X_train"], result["y_train"])
    y_test_pred = ols_predict(result["X_test"], theta)
    y_val_pred = ols_predict(result["X_val"], theta)
    eva_test = evaluate_predictions(result["y_test"], y_test_pred)
    eva_val = evaluate_predictions(result["y_val"], y_val_pred)

    return {
        "theta": theta,
        "y_test": result["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": eva_test,
        "val_metrics": eva_val
    }

