import time
import json
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    classification_report,
)
from sklearn.inspection import permutation_importance
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_iris,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor, TPOTClassifier
from pdpbox import pdp
import shap

# 제목 설정
st.title("AutoML Streamlit App")

# 데모 버전 설명 추가
st.info(
    """
**Note: This is a demo version with limited model performance.**
For the sake of faster execution and demonstration purposes:
- The number of generations and population size in the genetic algorithm are reduced.
- The maximum runtime for model search is restricted.
- The complexity of explored models is limited.

In a production environment, these constraints would be relaxed to potentially achieve better performance at the cost of longer computation time.
"""
)


# 문제 유형 선택
problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"])

# 데이터셋 선택
if problem_type == "Regression":
    dataset = st.selectbox("Select Dataset", ["California Housing", "Diabetes"])
    if dataset == "California Housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="MedHouseValue")
        df = pd.concat([df, target], axis=1)
        st.write("### California Housing Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains information about housing in California. The goal is to predict the median house value for California districts."
        )

    elif dataset == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="Disease Progression")
        df = pd.concat([df, target], axis=1)
        st.write("### Diabetes Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains information about diabetes patients. The goal is to predict a quantitative measure of disease progression one year after baseline."
        )


elif problem_type == "Classification":
    dataset = st.selectbox("Select Dataset", ["Iris", "Breast Cancer"])
    if dataset == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="Species")
        df = pd.concat([df, target], axis=1)
        st.write("### Iris Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains information about iris flowers. The goal is to classify iris flowers into three species based on their measurements."
        )

    elif dataset == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="Diagnosis")
        df = pd.concat([df, target], axis=1)
        st.write("### Breast Cancer Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal is to classify tumors as malignant or benign."
        )

# 히트맵 그리기
if dataset in ["Diabetes", "California Housing", "Iris", "Breast Cancer"]:
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    st.pyplot(plt)

# 타깃 변수의 히스토그램 그리기
if problem_type == "Regression":
    st.write("### Target Variable Histogram")
    plt.figure(figsize=(10, 6))
    plt.hist(target, bins=30, color="blue", alpha=0.7)
    plt.title(f"Distribution of {target.name}")
    plt.xlabel(target.name)
    plt.ylabel("Frequency")
    st.pyplot(plt)

elif problem_type == "Classification":
    st.write("### Target Variable Distribution")
    plt.figure(figsize=(10, 6))
    target.value_counts().plot(kind="bar")
    plt.title(f"Distribution of {target.name}")
    plt.xlabel(target.name)
    plt.ylabel("Count")
    st.pyplot(plt)

# AutoML 적용
if st.button("Run AutoML"):
    st.write("### Running AutoML...")

    # 데이터 분할
    X = df.drop(columns=[target.name])
    y = df[target.name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 진행 상황 표시를 위한 플레이스홀더
    progress_bar = st.progress(0)
    status_text = st.empty()

    # AutoML 모델 훈련
    start_time = time.time()
    if problem_type == "Regression":
        automl = TPOTRegressor(
            generations=5,
            population_size=20,
            cv=5,
            max_time_mins=5,
            max_eval_time_mins=0.5,
            verbosity=2,
            random_state=42,
        )
    else:  # Classification
        automl = TPOTClassifier(
            generations=5,
            population_size=20,
            cv=5,
            max_time_mins=5,
            max_eval_time_mins=0.5,
            verbosity=2,
            random_state=42,
        )

    # 훈련 진행 상황 업데이트
    for i in range(10):
        time.sleep(0.5)
        progress_bar.progress(i * 10)
        status_text.text(f"Training in progress... {i * 10}% complete")

    automl.fit(X_train, y_train)

    end_time = time.time()
    training_time = end_time - start_time

    # 결과 표시
    progress_bar.progress(100)
    status_text.text("Training complete!")
    st.write(f"AutoML completed in {training_time:.2f} seconds")

    st.write("### Model Summary")
    st.write(f"Best pipeline: {automl.fitted_pipeline_}")

    y_pred = automl.predict(X_test)

    if problem_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on test set: {accuracy:.4f}")
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))

        st.write("Interpretation:")
        st.write(
            "- Accuracy: Proportion of correct predictions (both true positives and true negatives) among the total number of cases examined."
        )
        st.write(
            "- Precision: Ability of the classifier not to label as positive a sample that is negative."
        )
        st.write(
            "- Recall: Ability of the classifier to find all the positive samples."
        )
        st.write("- F1-score: The harmonic mean of precision and recall.")

    else:  # Regression
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.write(f"R-squared score on test set: {r2:.4f}")
        st.write(f"Mean Squared Error (MSE) on test set: {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE) on test set: {rmse:.4f}")

        st.write("Interpretation:")
        st.write(
            "- R-squared: Proportion of the variance in the dependent variable that is predictable from the independent variable(s). Ranges from 0 to 1, where 1 indicates perfect prediction."
        )
        st.write(
            "- MSE: Average squared difference between the estimated values and the actual value."
        )
        st.write(
            "- RMSE: Square root of MSE. It's in the same unit as the target variable and represents the standard deviation of the residuals."
        )

    # 간단한 특성 중요도 (가능한 경우)
    if hasattr(automl.fitted_pipeline_[-1], "feature_importances_"):
        st.write("### Top 5 Important Features")
        importances = automl.fitted_pipeline_[-1].feature_importances_
        feature_imp = pd.DataFrame({"feature": X.columns, "importance": importances})
        feature_imp = feature_imp.sort_values("importance", ascending=False).head(5)
        st.table(feature_imp)
