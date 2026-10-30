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

