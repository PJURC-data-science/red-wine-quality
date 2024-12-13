from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


def two_sample_t_test(df, var_1, var_2):
    """
    Performs a two-sample t-test on the given DataFrame and returns the p-value.

    Args:
        df (DataFrame): Input DataFrame
        var_1 (str): Name of the variable to perform the t-test on
        var_2 (str): Name of the variable to perform the t-test on

    Returns:
        float: p-value
    """
    
    df_1 = df.loc[df[var_1] > df[var_1].mean()]
    df_2 = df.loc[df[var_1] <= df[var_1].mean()]

    t_stat, p_value = stats.ttest_ind(df_1[var_2], df_2[var_2])
    print(f"t-statistic: {t_stat:.6f}")
    print(f"p-value: {p_value:.6f}")

def calculate_weights(y_train):
    """
    Calculates class weights for the given training labels.

    Args:
        y_train (ndarray): Training labels

    Returns:
        dict: Dictionary containing class weights
    """

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    return class_weights_dict


def ordinal_logistic_regression(X_train, y_train, X_test, class_weights_dict):
    """
    Performs ordinal logistic regression on the training data and returns the predicted values for the test data.

    Args:
        X_train (ndarray): Training data features
        y_train (ndarray): Training data labels
        X_test (ndarray): Test data features
        y_test (ndarray): Test data labels
        
    Returns:
        ndarray: Predicted labels for the test data
    """
    
    # Create ordinal logistic regression object
    reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight=class_weights_dict)

    # Fit the model to the training data
    reg.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = reg.predict(X_test)

    return y_pred


def random_forest(X_train, y_train, X_test, class_weights_dict):
    """
    Performs random forest regression on the training data and returns the predicted values for the test data.

    Args:
        X_train (ndarray): Training data features
        y_train (ndarray): Training data labels
        X_test (ndarray): Test data features
        class_weights_dict (dict): Class weights dictionary

    Returns:
        ndarray: Predicted labels for the test data
    """

    # Define the Random Forest model
    rf_model = RandomForestClassifier(class_weight=class_weights_dict)

    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Apply Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Train the model with the best parameters
    best_rf_model = grid_search.best_estimator_
    best_rf_model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = best_rf_model.predict(X_test)
    
    # Extract feature importances
    feature_importances = best_rf_model.feature_importances_

    # Create a DataFrame for better visualization
    features = X_train.columns  # Assuming X_train is a DataFrame
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Print the feature importances
    print(importance_df)

    return y_pred


def get_model_results(y_test, y_pred):
    """
    Calculates the r-squared, mean squared error, and mean absolute error of the predicted values.

    Args:
        y_test (ndarray): True test labels
        y_pred (ndarray): Predicted labels

    Returns:
        dict: Dictionary containing the r-squared, mean squared error, and mean absolute error
    """
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Print the results
    print(f"R-squared: {r2}")
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")
    

def visualize_performance(y_test, y_pred):
    """
    Visualizes the model performance

    Args:
        y_test (ndarray): True test labels
        y_pred (ndarray): Predicted labels

    Returns:
        None
    """
    # 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))

    # Actual vs. predicted values
    sns.scatterplot(x=y_test, y=y_pred, ax=ax[0])
    ax[0].plot([1, 10], [1, 10], color='red', linestyle='--')
    ax[0].set_xlabel('Actual Quality Rating')
    ax[0].set_ylabel('Predicted Quality Rating')
    ax[0].set_title('Actual vs. Predicted Quality Rating')
    ax[0].set_xlim(1, 10)
    ax[0].set_ylim(1, 10)

    # Residuals
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=np.arange(min(residuals), max(residuals) + 2) - 0.5, ax=ax[1])
    ax[1].set_xlabel('Residuals')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Distribution of Residuals')
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Confusion Matrix
    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=unique_classes)
    cmd.plot(cmap=plt.cm.Blues, ax=ax[2])
    ax[2].set_title("Confusion Matrix")
    
    # Show plot
    plt.tight_layout()
    plt.show()