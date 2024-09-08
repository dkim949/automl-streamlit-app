import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_diabetes,
    fetch_california_housing,
    load_iris,
    load_breast_cancer,
)
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor, TPOTClassifier

# 제목 설정
st.title("AutoML Streamlit App")

# 문제 유형 선택
problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"])

# 데이터셋 선택
if problem_type == "Regression":
    dataset = st.selectbox("Select Dataset", ["Diabetes", "California Housing"])
    if dataset == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="Disease Progression")
        df = pd.concat([df, target], axis=1)
        st.write("### Diabetes Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains information about diabetes patients. The goal is to predict a quantitative measure of disease progression one year after baseline."
        )

    elif dataset == "California Housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name="MedHouseValue")
        df = pd.concat([df, target], axis=1)
        st.write("### California Housing Dataset")
        st.write(df.head())
        st.write(
            "**Overview**: This dataset contains information about housing in California. The goal is to predict the median house value for California districts."
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

    # AutoML 모델 훈련
    if problem_type == "Regression":
        automl = TPOTRegressor(
            generations=5, population_size=20, verbosity=2, random_state=42
        )
        automl.fit(X_train, y_train)
        st.write("Best pipeline:", automl.fitted_pipeline_)
        st.write("Score on test set:", automl.score(X_test, y_test))

    elif problem_type == "Classification":
        automl = TPOTClassifier(
            generations=5, population_size=20, verbosity=2, random_state=42
        )
        automl.fit(X_train, y_train)
        st.write("Best pipeline:", automl.fitted_pipeline_)
        st.write("Score on test set:", automl.score(X_test, y_test))

# 추가적인 분석 로직을 여기에 추가할 수 있습니다.
