from sklearn.pipeline import Pipeline
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE  # Import SMOTE


data = pd.read_csv('datasets/enhanced_feature_engineering_data.csv')


cols_to_impute_with_mean = ['prepayment', 'monthly_income', 'DTI_fraction']

for col in cols_to_impute_with_mean:
    data[col] = pd.to_numeric(data[col])
    data[col].fillna(data[col].mean(), inplace=True)
# df.method({'col': value}, inplace=True)
# print(data.isnull().sum().sort_values(ascending=False))

# Split data into features and target
X = data.drop(['EverDelinquent', 'prepayment'], axis=1)
y_class = data['EverDelinquent']
y_reg = data['prepayment']

# First I try to select the 10th best features
X_selected = X[['DTI', 'OrigUPB', 'OrigInterestRate', 'monthly_rate', 'monthly_payment',
                'total_payment', 'interest_amount', 'cur_principal', 'DTI_fraction',
                'monthly_income']]

# Split into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X_selected, y_class, y_reg, test_size=0.2, random_state=42
)


class CustomPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Classification pipeline
        self.classification_pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', SimpleImputer(strategy='median'),
                 X_selected.select_dtypes(include=['number']).columns)
            ])),
            ('clf', LogisticRegression(random_state=42))
        ])

        # Regression pipeline
        self.regression_pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', SimpleImputer(strategy='median'),
                 X_selected.select_dtypes(include=['number']).columns)
            ])),
            ('poly', PolynomialFeatures(degree=2)),  # Optional
            ('reg', LinearRegression())
        ])

    def fit(self, X_selected, y_class, y_reg):
        # Outlier detection
        iso = IsolationForest(contamination=0.01, random_state=42)
        y_outliers = iso.fit_predict(X_selected)
        X_filtered = X_selected[y_outliers == 1]
        y_class_filtered = y_class[y_outliers == 1]
        y_reg_filtered = y_reg[y_outliers == 1]

        # Apply SMOTE to handle class imbalance after outlier filtering
        smote = SMOTE(random_state=42)
        X_smote, y_class_smote = smote.fit_resample(
            X_filtered, y_class_filtered)

        # Fit classification pipeline on the balanced data
        self.classification_pipeline.fit(X_smote, y_class_smote)

        # Predict on full data
        y_class_pred = self.classification_pipeline.predict(X_selected)

        # Filter data based on classification predictions for regression
        X_filtered_reg = X_selected.loc[y_class_pred == 1]
        y_reg_filtered = y_reg.loc[y_class_pred == 1]

        # Ensure there are samples for regression
        if not X_filtered_reg.empty and not y_reg_filtered.empty:
            self.regression_pipeline.fit(X_filtered_reg, y_reg_filtered)
        else:
            raise ValueError("No samples for regression after filtering")
        return self

    def predict(self, X_selected):
        y_class_pred = self.classification_pipeline.predict(X_selected)
        print(f"Classification predictions: {y_class_pred}")

        # Filter data for regression prediction
        X_filtered_reg = X_selected.loc[y_class_pred == 1]
        print(f"Filtered data for regression: {X_filtered_reg.shape}")

        # If filtered data is empty, handle gracefully
        if X_filtered_reg.empty:
            print("No data to perform regression")
            return y_class_pred, np.array([])  # or return some default value

        # Predict regression if data exists
        y_reg_pred = self.regression_pipeline.predict(X_filtered_reg)
        return y_class_pred, y_reg_pred


# Create and fit the custom pipeline
pipeline = CustomPipeline()
pipeline.fit(X_train, y_class_train, y_reg_train)


# Save the pipeline
joblib.dump(pipeline, 'model/pipeline.pkl')

# Load and use the pipeline for predictions
# loaded_pipeline = joblib.load('combined_pipeline.pkl')
# y_class_pred, y_reg_pred = loaded_pipeline.predict(X_test)
