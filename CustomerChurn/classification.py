# 1. IMPORT DEPENDENCIES & DATA
# pandas-profiling: helps with data analysis
import pandas as pd

# - data explanation:
# "nps rating": Net promoter score is a market research metric that is based on a single survey question asking
# respondents to rate the likelihood that they would recommend a company, product, or a service to a friend.
df = pd.read_csv('./ChurnPrediction/CustomerChurn/classificationdata.csv', index_col='id')
# to show all columns
pd.set_option('display.max_columns', None)
print(df.head())
print(df.tail())


# %%
# DATA PREPROCESSING - 2. SPLIT DATA TO PREVENT SNOOPING BIAS
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=1234)

# axis=1 means dropping columns
X_temp = train.drop('churn', axis=1)

y_temp = train['churn']
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=1234)
print(y_train)


# %%
# 3. EXPLORATORY DATA ANALYSIS
# BIRDS EYE VIEW
from matplotlib import pyplot as plt
print(train.info())
print(train.isnull().sum())
print(train.describe())

# describe() function in pandas generates descriptive statistics of DataFrame or Series. When used with the
# parameter include='object', it will provide statistics only on the columns of type object, typically strings or
# categorical data.
print(train.describe(include='object'))
train.hist(figsize=(20, 20))
plt.show()

# FASTER EDA WITH PANDAS-PROFILING
# import pandas_profiling
# profile = pandas_profiling.ProfileReport(df=train, title='Pandas Profiling Report')
# profile.to_file(output_file="PandasProfilingReport.html")


# %%
# ANALYZE CATETORICAL FEATURES
# import target features
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.countplot(x='churn', data=train).set_title('Churn vs Non-Churn Classes')
plt.show()

# inspect state, area code and promotion offered
print(train.dtypes)
fig, axs = plt.subplots(3, figsize=(25, 15))
sns.countplot(data=train, x='state_code', ax=axs[0])
sns.countplot(data=train, x='promotions_offered', ax=axs[1])
sns.countplot(data=train, x='area_code', ax=axs[2])
plt.show()
# save plot to disc
# fig.get_figure().savefig('CustomersChurn_CateFeatures.png')

# %%
# ANALYZE NUMERICAL FEATURES
# inspect Vmail messages and Customer Service Calls
print(train.dtypes)

fig, axs = plt.subplots(2, figsize=(25, 10))
# Adjust the space between subplots
plt.subplots_adjust(hspace = 0.4)
sns.countplot(data=train, x='number_vmail_messages', ax=axs[0]).set_title('Voice Mail Messages')
sns.countplot(data=train, x='number_customer_service_calls', ax=axs[1]).set_title('Customer Service Calls')
plt.show()
# save plot to disc
# fig.get_figure().savefig('CustomersChurn_NumFeatures.png')

# %%
# analyze customer service calls given it's skewed
import numpy as np
print(train['number_customer_service_calls'].skew())
# a skewness value > 0 means that there is more weight in the left tail of the distribution.
train['log_customer_service_calls'] = np.log(train['number_customer_service_calls'] + 1)
print(train['log_customer_service_calls'].skew())
# closer to 0, less skewed
train[['number_customer_service_calls', 'log_customer_service_calls']].hist(figsize=(25, 5))
plt.show() # shows that the log transformation has made the distribution more normal
# since we are still exploring, we need to drop the log_customer_service_calls column
train.drop(['log_customer_service_calls'], axis=1)
print(train.dtypes)


# %%
# ANALYZE RELATIONSHIP BETWEEN FEATURES
# look into numeric/numeric correlation
sns.heatmap(train.select_dtypes(exclude='object').corr()).set_title('Correlation Plot')
plt.show()
print(train.dtypes)
temp = train.copy()
temp['churn'] = temp['churn'].apply(lambda x: 1 if x == 'yes' else 0)
sns.heatmap(temp.select_dtypes(exclude='object').corr()).set_title('Correlation Plot')
plt.show()


# plot churn distribution
# plot out the distributions against churn
for col in train.select_dtypes(exclude='object').columns:
    sns.violinplot(x='churn', y=col, data=train).set_title(f'Churn vs {col}')
    plt.show()

# for col in train.select_dtypes(exclude='object').columns:
#     print(col)
# since we found these two features maybe highly related to the churn
plt.title('NPS Rating vs Remaining Term')
sns.barplot(x='last_nps_rating', y='remaining_term', hue='churn', data=train)
plt.show()


# %%
# Categorical Relationships to Churn
# display unique features
print(train.promotions_offered.unique())
# replace the weird ones with "No"pd.crosstab(train[pivot_feature], train['churn']) / len(train)
train['promotions_offered'] = train['promotions_offered'].replace(['NO', np.NaN], 'No')
print(train.promotions_offered.unique())
# pivot
pivot_feature = 'promotions_offered'
print(pd.crosstab(train[pivot_feature], train['churn']) / len(train))
# loop through all categorical features
for col in train.select_dtypes(include='object').columns:
    if col != 'churn':
        print(pd.crosstab(train[col], train['churn']) / len(train))


# %%
# relationships between remaining term and NPS Rating? and Promotions Offered?
# we picked 7 and 5 based on the stats graph
print(train[['remaining_term']].head())
train['nps_less_7'] = (train['last_nps_rating'] < 7).astype(int)
train['remaining_term_less_5'] = (train['remaining_term'] < 5).astype(int)
train['churn_yes'] = train['churn'].apply(lambda x: 1 if x == 'yes' else 0)
train['churn_no'] = train['churn'].apply(lambda x: 0 if x == 'no' else 1)
print(train.head())
# group by method
group_nps_term = train.groupby(['nps_less_7', 'remaining_term_less_5']).sum(['churn_yes', 'churn_no'])[['churn_yes', 'churn_no']] / len(train)
print(group_nps_term)
group_nps_term_promo = train.groupby(['nps_less_7', 'remaining_term_less_5', 'promotions_offered']).sum(['churn_yes', 'churn_no'])[['churn_yes', 'churn_no']] / len(train)
print(group_nps_term_promo)
# since we are still in the EDA process
train.drop(['nps_less_7', 'remaining_term_less_5', 'churn_yes', 'churn_no'], axis=1, inplace=True)
print(train.columns)



# %%
# 4. DATA PREPROCESSING
# double check if we're operating with a clean slate
print(train.columns)
print(df.columns)
print([col in df.columns for col in df.columns])
print(train.isnull().sum())

# missing area code data - categorical
print(train[train['area_code'].isnull()].head())
train['area_code'] = train['area_code'].fillna('missing')
print(train[train['area_code'] == 'missing'])
print(train.isnull().sum())

# missing voice mail plan - categorical
print(train.dtypes)
train['voice_mail_plan'] = train['voice_mail_plan'].fillna('missing')
print(train[train['voice_mail_plan'] == 'missing'])

# missing Evening Minutes - numeric
print(train[train['total_eve_minutes'].isnull()][['total_eve_minutes', 'churn']])
mean_eve_mins = train['total_eve_minutes'].mean().round(2)
print(mean_eve_mins)
print(train['total_day_charge'].std())
train['total_eve_minutes_missing'] = train['total_eve_minutes'].isnull().astype(int)
train['total_eve_minutes'] = train['total_eve_minutes'].fillna(mean_eve_mins)
print(train[train['total_eve_minutes_missing'] == 1][['total_eve_minutes_missing', 'total_eve_minutes']])

# missing target variables
print(train.isnull().sum())
print(train[train['churn'].isnull()])

# the only NaNs left have same length, we can now drop them
train = train[~train['churn'].isnull()]
print(train.isnull().sum())



# %%
# 5. FEATURE ENGINEERING
# building a ratio for correlated predictors
sns.heatmap(train.select_dtypes(exclude='object').corr())
plt.show()
train['day_ratio'] = train['total_day_charge'] / train['total_day_minutes']
train['eve_ratio'] = train['total_eve_charge'] / train['total_eve_minutes']
train['night_ratio'] = train['total_night_charge'] / train['total_night_minutes']
train['intl_ratio'] = train['total_intl_charge'] / train['total_intl_minutes']
print(train.head())
train = train.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge',
                    'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'], axis=1)
print(train.columns)

# skewed customer service calls
train['number_customer_service_calls'] = np.log(train['number_customer_service_calls']+1) #+1 prevents the divide by 0 error
train['number_customer_service_calls'].hist(figsize=(20, 5))
plt.show()

# feature engineering unhappy customers
train['promotions_offered'] = train['promotions_offered'].replace(['NO', np.NaN], 'No')
print(train.isnull().sum())
train['unhappy_customer'] = ((train.remaining_term < 5) & (train.last_nps_rating <= 7) &
                             (train.promotions_offered == 'No')).astype(int)
# print(train[train['remaining_term'] < 5][['remaining_term', 'last_nps_rating', 'promotions_offered', 'unhappy_customer']])
print(train[train['unhappy_customer'] == 1].head())


# %%
# create target and feature values
from sklearn.preprocessing import OneHotEncoder
# create X and y variables
X_train = train.drop(['churn'], axis=1)
y_train = np.where(train['churn'] == 'yes', 1, 0)
# validating churn is dropped
print('churn' in list(X_train.columns))
# check y value
print(y_train)

# create the one hot encoder
onehot = OneHotEncoder(handle_unknown='ignore')
# apply one hot encoding to categorical columns
# fitting the OneHotEncoder to the categorical columns in X_train (those with data type 'object'),
# transforming those columns into one-hot encoded arrays, and converting the result to a dense array.
encoded_columns = onehot.fit_transform(X_train.select_dtypes(include='object')).toarray()
print(encoded_columns)
print(onehot.get_feature_names_out())
# replacing X_train with a new DataFrame that includes only the columns of X_train that do not have the 'object'
# data type. The 'object' columns are excluded temporarily because they are being one-hot encoded.
X_train = X_train.select_dtypes(exclude='object')
# After one-hot encoding, they are added back to X_train with the following line:
X_train[onehot.get_feature_names_out()] = encoded_columns
print(X_train.columns)


# %%
# DEALING WITH IMBALANCED CLASSES
# since only a small chunck of data has the "yes" as the value for churn, it is imbalanced
# we need to balance the data
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# create the SMOTE class
sm = SMOTE(random_state=1234)
# can also use RandomOverSampler and RandomUnderSampler but the Undersampler will reduce the data
# Resample to balance the dataset
X_train, y_train = sm.fit_resample(X_train, y_train)
sns.countplot(x=y_train).set_title('Balanced Dataset')
plt.show()


# %%
# 6. MODELING
# MODELING - build pipelines
# import pipeline dependencies
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# import algorithms
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
pipelines = {
    'sgd': make_pipeline(StandardScaler(), SGDClassifier()),
    'ridge': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'xg': make_pipeline(StandardScaler(), XGBClassifier()),
}

# MODELING - build grids
print(pipelines['xg'].get_params())
grid = {
    'sgd':{
        'sgdclassifier__alpha':[0.00001, 0.0001, 0.001, 0.01]
    },
    'ridge':{
        'ridgeclassifier__alpha':[0.01, 0,5, 1.0, 2.0, 3.0]
    },
    'rf':{
        'randomforestclassifier__n_estimators':[50,100, 200,300],
        'randomforestclassifier__max_depth':[None, 5, 7, 9]
    },
    'gb':{
        'gradientboostingclassifier__n_estimators':[50,100, 200,300],
        'gradientboostingclassifier__max_depth':[None, 3, 5, 7, 9]
    },
    'xg':{
        'xgbclassifier__n_estimators':[50,100, 200,300],
        'xgbclassifier__max_depth':[None, 3, 5, 7, 9]
    }
}

# %%
# MODELING - train models
from sklearn.model_selection import GridSearchCV
fit_models = {}
for algo, pipeline in pipelines.items():
    try:
        # print(algo)
        print(f'Commencing training for {algo}')
        # Training the model
        model = GridSearchCV(pipeline, grid[algo], cv=10, n_jobs=-1)
        model.fit(X_train, y_train)
        fit_models[algo] = model
        print(f'training completed for {algo}')

    except Exception as e:
        print(f'There was an error with training the {algo} model: {e})')


# %%
print(fit_models)

# %%
# 7. EVALUATION
# apply transformations to test data
col_order = X_train.columns
print(col_order)


def transform_data(test_df, col_order, mean_eve_mins, onehot):
    # copy data frame
    X = test_df.copy()

    # DP(data processing) - Handle Mising Values
    X['area_code'] = X['area_code'].fillna('missing')
    X['voice_mail_plan'] = X['voice_mail_plan'].fillna('missing')
    X['total_eve_minutes_missing'] = X['total_eve_minutes'].isnull().astype(int)
    X['total_eve_minutes'] = X['total_eve_minutes'].fillna(mean_eve_mins)

    # FE(feature engineering) - Ratios
    X['day_ratio'] = X['total_day_charge'] / X['total_day_minutes']
    X['eve_ratio'] = X['total_eve_charge'] / X['total_eve_minutes']
    X['night_ratio'] = X['total_night_charge'] / X['total_night_minutes']
    X['intl_ratio'] = X['total_intl_charge'] / X['total_intl_minutes']
    X = X.drop(['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge',
                        'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes'], axis=1)

    # FE - Log Transformations
    X['log_customer_service_calls'] = np.log(X['number_customer_service_calls'] + 1)

    # FE - Unhappy Customer
    X['promotions_offered'] = X['promotions_offered'].replace(['NO', np.NaN], 'No')
    X['unhappy_customer'] = ((X.remaining_term < 5)
                             & (X.last_nps_rating <= 7)
                             & (X.promotions_offered == 'No')).astype(int)

    # Onehot Encoder
    # we have alreay fit the transformation on our test data set, so we dont need to fit it again, so instead of using
    # .fit_transform, we use .transform
    encoded_columns = onehot.transform(X.select_dtypes(include='object')).toarray()
    X = X.select_dtypes(exclude='object')
    X[onehot.get_feature_names_out()] = encoded_columns

    return X[col_order]


test = test[~test['churn'].isnull()]
print(test.isnull().sum())
X_test = transform_data(test.drop('churn', axis=1), col_order, mean_eve_mins, onehot)
print(X_test)
y_test = np.where(test['churn'] == 'yes', 1, 0)
print(y_test[:50])


# %%
# EVALUATION - EVALUATE PERFORMANCE METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
for algo, model in fit_models.items():
    # make a predictions
    yhat = model.predict(X_test)
    # calculating metrics test partion and our predictions
    accuracy = accuracy_score(y_test, yhat)
    precision = precision_score(y_test, yhat)
    recall = recall_score(y_test, yhat)
    # print it out
    print(f'{algo} model scores Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}')


# %%
# UNDERSTANDING PERFORMANCE METRICS
model = fit_models['gb']
yhat = model.predict(X_test)
print(yhat[:50])
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# [[ 644  97]
#  [ 173  4241]]

# %%
accuracy = 4241 / (644+97+173+4241)
# accuracy = 4156 / (669+72+258+4156)
print(accuracy)
precision = 644 / (644+173)
# precision = 669 / (669+258)
print(precision)
recall = 644 / (644+97)
# recall = 669 / (669+72)
print(recall)


# %%
# MAKE A PREDICTION
yhat = model.predict(X_test)
print(yhat)

# %%
# create dataframe
res = pd.DataFrame([y_test, yhat])
res = res.T
res.columns = ['ytrue', 'ypred']
print(res.head(8))
print(X_test.iloc[5])


# %%
# 8. DEPLOYMENT
# save models and encoder
import os
import pickle
# create file paths
SAVE_PATH = os.path.join('models', 'experiment_1')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# %%
# save machine learning models
for algo, fit_models in fit_models.items():
    FILE_PATH = os.path.join(SAVE_PATH, f'{algo}.pkl')
    with open(FILE_PATH, 'wb') as f:
        pickle.dump(model, f)


# %%
# save one hot encoder
ENCODER_FILE_PATH = os.path.join(SAVE_PATH, 'encoder.pkl')
with open(ENCODER_FILE_PATH, 'wb') as f:
    pickle.dump(onehot, f)


# %%
# DEPLOYMENT - CREATE A MODEL SCHEMA
def extract_column_values(col, df):
    if df[col].dtype == 'O':
        return list(df[col].unique())
    elif df[col].dtype == 'int64':
        min = int(df[col].min())
        max = int(df[col].max())
        return (min, max)
    elif df[col].dtype == 'float64':
        min = float(df[col].min())
        max = float(df[col].max())
        return (min, max)
    else:
        return list(df[col].unique().astype(str))

# %%
# use the function to get the values from all the columns in that dataframe
column_info = {
    col: {'dtype': str(df[col].dtype),
          'values': extract_column_values(col, df)}
    for col in df.columns
}
print(column_info)



# %%
transformed_cols = {'transformed_columns': X_train.columns.to_list()}
schema = {'column_info': column_info, 'transformed_columns': transformed_cols}
print(transformed_cols)
print(schema)


# %%
# create a path
os.makedirs('app')

# %%
# save the schema to an json
import json
with open(os.path.join('app', 'schema.json'), 'w') as f:
    json.dump(schema, f)

# %%
# 9.TEST SCORING
# load models and encoder

# load ML Model
with open(os.path.join(SAVE_PATH, 'gb.pkl'), 'rb') as f:
    model = pickle.load(f)
print(model)
# load Encoder
with open(os.path.join(SAVE_PATH, 'encoder.pkl'), 'rb') as f:
    onehot = pickle.load(f)
# load schema
with open(os.path.join('app', 'schema.json'), 'r') as f:
    schema = json.load(f)


# %%
# transform test sample and predict
res = {
  "state_code": "OH",
  "area_code": "area_code_415",
  "international_plan": "no",
  "voice_mail_plan": "yes",
  "number_vmail_messages": "26",
  "total_day_minutes": 161.6,
  "total_day_calls": 123,
  "total_day_charge": 27.47,
  "total_eve_minutes": 195.5,
  "total_eve_calls": 103,
  "total_eve_charge": 16.62,
  "total_night_minutes": 254.4,
  "total_night_calls": 103,
  "total_night_charge": 11.45,
  "total_intl_minutes": 13.7,
  "total_intl_calls": 3,
  "total_intl_charge": 3.7,
  "promotions_offered":"No",
  "number_customer_service_calls": 5.0,
  "tenure":4,
  "contract_length":5,
  "remaining_term":10,
  "last_nps_rating":10
}

# %%
# Extract column orders
column_order_in = list(schema['column_info'].keys())[:-1]
column_order_out = list(schema['transformed_columns']['transformed_columns'])


# %%
# Convert single prediction to a DF
scoring_data = pd.Series(res).to_frame().T
scoring_data = scoring_data[column_order_in]
print(scoring_data)


# %%
# Check datatypes
for column, column_properties in schema['column_info'].items():
    if column != 'churn':
        dtype = column_properties['dtype']
        scoring_data[column] = scoring_data[column].astype(dtype)


# %%
scoring_sample = transform_data(scoring_data, column_order_out, mean_eve_mins, onehot)
print(model.predict(scoring_sample))
print(mean_eve_mins)
