import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('datasets/LoanExport.csv', low_memory=False)
# data.info()


# monthly interest rate
data['OrigInterestRate_Monthly'] = np.round(
    (data['OrigInterestRate'] / 12) / 100, 4)

# monthly installment


def calculateEmi(principal, monthly_interest_rate, loan_term_months):
    numerator = (1 + monthly_interest_rate) ** loan_term_months
    denominator = numerator - 1
    interest = numerator / denominator
    emi = principal * monthly_interest_rate * interest
    return np.int64(emi)


data['MonthlyInstallment'] = data.apply(
    lambda features: calculateEmi(
        principal=features['OrigUPB'],
        monthly_interest_rate=features['OrigInterestRate_Monthly'],
        loan_term_months=features['OrigLoanTerm']), axis=1)

# current unpaid principal


def get_currentUPB(principal, monthly_interest_rate, monthly_installment,
                   payments_made):
    monthly_interest = monthly_interest_rate * principal
    monthly_paid_principal = monthly_installment - monthly_interest
    unpaid_principal = principal - (monthly_paid_principal * payments_made)
    return np.int32(unpaid_principal)


data['CurrentUPB'] = data.apply(
    lambda features: get_currentUPB(
        monthly_interest_rate=features['OrigInterestRate_Monthly'],
        principal=features['OrigUPB'],
        monthly_installment=features['MonthlyInstallment'],
        payments_made=features['MonthsInRepayment']), axis=1)

# monthly income


def calculate_monthly_income(dti, emi):
    dti = dti if dti < 1 else dti / 100
    # Calculate montly income
    if dti == 0:
        monthly_income = emi
    else:
        monthly_income = emi / dti
    return np.int64(monthly_income)


data['MonthlyIncome'] = data.apply(
    lambda features: calculate_monthly_income(
        dti=features['DTI'],
        emi=features['MonthlyInstallment']), axis=1)

# prepayment


def calculatePrepayment(dti, monthly_income):
    if (dti < 40):
        prepayment = monthly_income / 2
    else:
        prepayment = monthly_income * 3 / 4
    return np.int64(prepayment)


data['Prepayment'] = data.apply(
    lambda features: calculatePrepayment(
        dti=features['DTI'],
        monthly_income=features['MonthlyIncome']), axis=1)
data['Prepayment'] = (data['Prepayment']*24)-(data['MonthlyInstallment']*24)

# total payment and interest amount
data['Totalpayment'] = data['MonthlyInstallment'] * data['OrigLoanTerm']
data['InterestAmount'] = data['Totalpayment'] - data['OrigUPB']


le = LabelEncoder()
cat_col = ['FirstTimeHomebuyer', 'PPM', 'NumBorrowers', 'LoanSeqNum', 'PropertyState',
           'ProductType', 'ServicerName', 'PropertyType', 'Channel', 'SellerName']
data[cat_col] = data[cat_col].apply(le.fit_transform)

one_col = ['LoanPurpose', 'Occupancy']
data_one = pd.get_dummies(data[one_col], drop_first=True)
data_one = data_one.astype(int)

data = pd.concat([data, data_one], axis=1)
data.drop(['LoanPurpose', 'Occupancy'], inplace=True, axis=1)

# Split data into features and target
X = data.drop(['EverDelinquent', 'Prepayment', 'MSA', 'PostalCode'], axis=1)
y_class = data['EverDelinquent']
y_reg = data['Prepayment']

# Split into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)


class CustomPipelineWithFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, clf, reg, clf_features, reg_features):
        self.clf = clf
        self.reg = reg
        self.clf_features = clf_features
        self.reg_features = reg_features
        self.scaler_clf = StandardScaler()
        self.scaler_reg = StandardScaler()

    def fit(self, X, y_class, y_reg):
        # Ensure X is a DataFrame and contains the specified features
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")

        # Extract features for classification and scale them
        X_clf = X[self.clf_features]
        X_clf_scaled = self.scaler_clf.fit_transform(X_clf)
        self.clf.fit(X_clf_scaled, y_class)

        # Filter data where classification is 1
        X_filtered = X[y_class == 1]
        y_reg_filtered = y_reg[y_class == 1]

        # Extract features for regression and scale them
        X_filtered_reg = X_filtered[self.reg_features]
        X_filtered_reg_scaled = self.scaler_reg.fit_transform(X_filtered_reg)
        self.reg.fit(X_filtered_reg_scaled, y_reg_filtered)
        return self

    def predict(self, X):
        # Ensure X is a DataFrame and contains the specified features
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")

        # Extract features for classification and scale them
        X_clf = X[self.clf_features]
        X_clf_scaled = self.scaler_clf.transform(X_clf)
        y_class_pred = self.clf.predict(X_clf_scaled)

        # Initialize predictions for regression with NaNs
        y_reg_pred = np.full(X.shape[0], np.nan)

        # Filter data where classification is 1
        X_filtered = X[y_class_pred == 1]
        if len(X_filtered) > 0:
            # Extract features for regression and scale them
            X_filtered_reg = X_filtered[self.reg_features]
            X_filtered_reg_scaled = self.scaler_reg.transform(X_filtered_reg)
            y_reg_pred_filtered = self.reg.predict(X_filtered_reg_scaled)
            # Assign regression predictions to corresponding positions
            y_reg_pred[y_class_pred == 1] = y_reg_pred_filtered
        return y_class_pred, y_reg_pred


# Define feature sets
clf_features = ['CreditScore', 'Units', 'PropertyType', 'OrigLoanTerm',
                'MonthsDelinquent', 'MonthsInRepayment', 'Occupancy_O']

reg_features = ['MonthlyIncome', 'InterestAmount', 'Totalpayment', 'MonthlyInstallment',
                'OrigUPB', 'CurrentUPB', 'DTI', 'LoanSeqNum', 'NumBorrowers', 'FirstTimeHomebuyer']
# Create and fit the custom pipeline with Random Forest Regressor
pipeline = CustomPipelineWithFeatureSelection(
    clf=GaussianNB(),
    reg=RandomForestRegressor(n_estimators=100, max_depth=10,
                              min_samples_split=5, min_samples_leaf=5, random_state=42),
    clf_features=clf_features,
    reg_features=reg_features
)

# Fit the pipeline
pipeline.fit(X_train, y_class_train, y_reg_train)

# Make predictions
y_class_pred, y_reg_pred = pipeline.predict(X_test)

# print("Predictions class:", y_class_pred)
# print("Predictions reg:", y_reg_pred)


# Save the pipeline
joblib.dump(pipeline, 'model/pipeline.pkl')
# print(data['FirstTimeHomebuyer'])
