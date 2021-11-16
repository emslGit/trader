import os, requests, json
from numpy.random.mtrand import randint
from matplotlib.pyplot import annotate, pcolor, spy
from textwrap import indent
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import spdiags

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report

base_url = 'https://cloud.iexapis.com/v1'
sandbox_url = 'https://sandbox.iexapis.com/v1'

token = os.environ.get('IEX_TOKEN')
# token = 'Tsk_3339c0f8d2e34f4db867c257fcffa153'
params = {
    'token': token,
    'types': 'chart',
    'range': '0y'
}

## Common Indices
# SPY - SP500
# DIA - DJIA
# DAX - DAX
# FEZ - STOXX50
# AIA - ASIA50

## By Cap Size
# EEM - Emerging markets
# MDY - Mid Cap
# IWM - Smallcap

## FX
# FXE - Euro Index
# UUP - Dollar Index
# GLD - Gold

## Bonds
# HYG - High Yield
# IEF - US10YR (inverted)
# TLT - US20YR (inverted?)

## Other Indicators
# VXX - VIX Short Term
# VIXM - VIX Mid Term

## Sectors
# TAN - Green Energy
# XLE - Brown Energy
# XLF - Finance
# XLI - Industrial
# XLP - Consumer Staples
# DBC - Commodities
# DBB - Base Metals

# symbols = 'SPY,DIA,FEZ,EEM,MDY,IWM,FXE,UUP,GLD,HYG,IEF,TLT,VIXM,XLE,XLF,XLI,XLP,DBC,DBB,AIA,DAX,TAN,VXX'.split(',')
symbols = 'SPY,DIA,FEZ,EEM,MDY,IWM,FXE,UUP,GLD,HYG,IEF,TLT,VIXM,XLE,XLF,XLI,XLP,DBC,DBB'.split(',')


###### POPULATE ######
# symbols_count = len(symbols)

# buf = list()
# for i, sym in enumerate(symbols):
#     print(f'{sym}: symbol {i + 1} out of {symbols_count}...')

#     res = requests.get(base_url + f'/stock/{sym}/chart', params=params)
#     res.raise_for_status()
#     buf += res.json()


# with open('./SPY_1.json', 'w') as f:
#     f.write(json.dumps(buf, indent=2))


daily = list()
###### PANDAS ######
with open('./price_data.json', 'r') as f:
    json_data = json.loads(f.read())
    
    for entry in json_data:
        date = entry['date'].split('-')
        week = datetime.date(int(date[0]), int(date[1]), int(date[2])).isocalendar()[1]
        daily.append({
            "symbol": entry['symbol'],
            "open": entry['open'],
            "close": entry['close'],
            "high": entry['high'],
            "low": entry['low'],
            "volume": entry['volume'],
            "change": entry['change'],
            "changePercent": entry['changePercent'],
            "date": entry['date'],
            "week": week,
            "anno": date[0],
            "month": int(date[1]),
        })

pd.set_option('display.width', 1920)
daily = pd.DataFrame(daily)
daily.sort_values(by = ['symbol','date'], inplace = True)


# weekly = list()
# i = 0
# for group in daily.groupby(['symbol', 'anno', 'week']):
#     weekly.append({
#         "symbol": list(group[1]['symbol'])[0],
#         "anno": list(group[1]['anno'])[0],
#         "week": list(group[1]['week'])[0],
#         "open": list(group[1]['open'])[0],
#         "close": list(group[1]['close'])[-1],
#         "high": group[1]['high'].max(),
#         "low": group[1]['low'].min(),
#         "volume": group[1]['volume'].sum()
#     })

# weekly = pd.DataFrame(weekly)


spyder = daily[daily['symbol'] == 'SPY']


###### SMOOTH ######
# out = 20

# smoothed = spyder.groupby('symbol')[['close', 'low', 'high', 'open', 'volume']].transform(lambda x: x.ewm(span = out).mean())

# spyder = pd.concat([spyder[['symbol', 'date', 'month', 'week', 'anno']], smoothed], axis=1, sort=False)

# flag = spyder.groupby('symbol')['close'].transform(lambda x : np.sign(x.diff(out)))
# spyder = spyder.assign(flag = flag)


###### OTHER INDICES ######
# spyder = spyder.assign(change = spyder['close'].diff())

for sym in symbols[1:]:
    spyder = spyder.assign(**{sym: daily[daily['symbol'] == sym]['close'].values})

# spyder = spyder.assign(TLT_IWM = spyder['TLT']/spyder['IWM'])


###### RSI ######
n = 25

# First make a copy of the data frame twice
up_df, down_df = spyder[['symbol','change']].copy(), spyder[['symbol','change']].copy()

# For up days, if the change is less than 0 set to 0.
up_df.loc['change'] = up_df.loc[(up_df['change'] < 0), 'change'] = 0

# For down days, if the change is greater than 0 set to 0.
down_df.loc['change'] = down_df.loc[(down_df['change'] > 0), 'change'] = 0

# We need change in price to be absolute.
down_df['change'] = down_df['change'].abs()

# Calculate the EWMA (Exponential Weighted Moving Average), meaning older values are given less weight compared to newer values.
ewma_up = up_df.groupby('symbol')['change'].transform(lambda x: x.ewm(span = n).mean())
ewma_down = down_df.groupby('symbol')['change'].transform(lambda x: x.ewm(span = n).mean())

# Calculate the Relative Strength
relative_strength = ewma_up / ewma_down

# Calculate the Relative Strength Index
relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

# Add the info to the data frame.
spyder = spyder.assign(rsi = relative_strength_index)


###### STOCHASTIC OSCILLATOR ######
n = 14

# Make a copy of the high and low column.
low, high = spyder[['symbol','low']].copy(), spyder[['symbol','high']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low = low.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
high = high.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

k_percent = 100 * ((spyder['close'] - low) / (high - low))
spyder = spyder.assign(k_percent = k_percent)


###### WILLIAMS R% ######
n = 14

# Make a copy of the high and low column.
low, high = spyder[['symbol','low']].copy(), spyder[['symbol','high']].copy()

# Group by symbol, then apply the rolling function and grab the Min and Max.
low = low.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
high = high.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

r_percent = ((high - spyder['close']) / (high - low)) * - 100
spyder = spyder.assign(r_percent = r_percent)


###### MOVING AVG ######
ema_tiny = spyder.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 30).mean())
ema_short = spyder.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 60).mean())
ema_mid = spyder.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 120).mean())
ema_long = spyder.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 200).mean())
macd_short = ema_tiny - ema_short
macd_long = ema_short - ema_long

spyder = spyder.assign(macd_short = macd_short)
spyder = spyder.assign(macd_long = macd_long)
spyder = spyder.assign(ema_tiny = ema_tiny.values)
spyder = spyder.assign(ema_short = ema_short.values)
spyder = spyder.assign(ema_mid = ema_mid.values)
spyder = spyder.assign(ema_long = ema_long.values)


###### PREPARATION ######
DAYS = 15

def stuff(val):
    if val >= 1.0325:
        return '+'
    elif val < 1.0325:
        return '~'
    else:
        return np.NAN

close_groups = spyder.groupby('symbol')['close']
close_groups = close_groups.transform(lambda x : x.shift(-DAYS) / x)
close_groups = close_groups.map(lambda x: stuff(x))

spyder = spyder.assign(prediction = close_groups)

print('Before NaN Drop we have {} rows and {} columns'.format(spyder.shape[0], spyder.shape[1]))
spyder = spyder.dropna()
print('After NaN Drop we have {} rows and {} columns'.format(spyder.shape[0], spyder.shape[1]))

present = spyder[spyder['anno'] != '2021']
present = present[present['anno'] != '2020']
# present = present[present['anno'] != '2019']
future = spyder[spyder['anno'] == '2020']
future = future.append(spyder[spyder['anno'] == '2021'])
# future = future.append(spyder[spyder['anno'] == '2021'])

future = future[randint(0, 100):randint(np.shape(future)[0] - 100, np.shape(future)[0])]


###### TRAINING ######
inputs = ['close', 'rsi', 'HYG', 'TLT', 'VIXM', 'ema_long'] #'IEF', 'UUP', 'MDY', 'FEZ', 'GLD', 'HYG', 'VIXM', 'XLF', 'XLI', 'DBC']
X_train = present[inputs]
y_train = present['prediction']
X_test = future[inputs]
y_test = future['prediction']

# Create a Random Forest Classifier
rand_frst_clf = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=180, min_samples_split=40, oob_score = True, criterion = "gini")

# Fit the data to the model
rand_frst_clf.fit(X_train, y_train)

# Make predictions
y_pred = rand_frst_clf.predict(X_test)


###### EVALUATION ######
# Print the Accuracy of our Model.
# print('Correct Prediction (%): ', accuracy_score(y_test, rand_frst_clf.predict(X_test), normalize = True) * 100.0)
# print('Random Forest Out-Of-Bag Error Score: {}'.format(rand_frst_clf.oob_score_))

# Build a classifcation report
report = classification_report(y_true = y_test, labels = ['+', '~'], y_pred = y_pred, output_dict = True)

# Add it to a data frame, transpose it for readability.
report_df = pd.DataFrame(report).transpose()
print(f"\n{report_df}")


###### FEATURE IMPORTANCE ######
feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(f"\n{feature_imp}")

# store the values in a list to plot.
# x_values = list(range(len(rand_frst_clf.feature_importances_)))

# # Cumulative importances
# cumulative_importances = np.cumsum(feature_imp.values)

# # Make a line graph
# plt.plot(x_values, cumulative_importances, 'g-')

# # Draw line at 95% of importance retained
# plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')

# # Format x ticks and labels
# plt.xticks(x_values, feature_imp.index, rotation = 'vertical')

# # Axis labels and title
# plt.xlabel('Variable')
# plt.ylabel('Cumulative Importance')
# plt.title('Random Forest: Feature Importance Graph')
# plt.show()


###### CONFUSION MATRIX ######
# rf_matrix = confusion_matrix(y_test, y_pred)

# true_negatives = rf_matrix[0][0]
# false_negatives = rf_matrix[1][0]
# true_positives = rf_matrix[1][1]
# false_positives = rf_matrix[0][1]

# accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
# percision = true_positives / (true_positives + false_positives)
# recall = true_positives / (true_positives + false_negatives)
# specificity = true_negatives / (true_negatives + false_positives)

# print('Accuracy: {}'.format(float(accuracy)))
# print('Percision: {}'.format(float(percision)))
# print('Recall: {}'.format(float(recall)))
# print('Specificity: {}'.format(float(specificity)))

# disp = plot_confusion_matrix(rand_frst_clf, X_test, y_test, display_labels = ['~', '+'], normalize = 'true', cmap=plt.cm.Blues)
# disp.ax_.set_title('Confusion Matrix - Normalized')
# plt.show()


##### RECIEVER OPERATING CHARACTERISTIC ######
fig, ax = plt.subplots()

# Create an ROC Curve plot.
rfc_disp = plot_roc_curve(rand_frst_clf, X_test, y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)

# Add our Chance Line
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

# Make it look pretty.
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")

# Add the legend to the plot
ax.legend(loc="lower right")

plt.show()

###### TESTING RATIOS ######
# target = spyder['close'].shift(-20) / spyder['close']
# syms = ['close', 'month', 'week', 'changePercent', 'volume', 'rsi', 'k_percent', 'r_percent', 'macd_short', 'macd_long', 'ema_tiny', 'ema_short', 'ema_mid', 'ema_long'] + symbols[2:]

# for i in range(len(syms)):
#     j = 0
#     print(syms[i])
#     plt.bar(target, spyder[syms[i]])
#     plt.show()    
#     plt.bar(spyder[syms[i]], target)
#     plt.show()
#     continue

#     while j < i:
#         print(syms[i], syms[j])
#         plt.scatter(spyder[syms[i]] / spyder[syms[j]], target)
#         plt.show()
#         j += 1

# plt.plot(spyder['ema_long'].values, spyder['close'].shift(-20))


# ###### IMPROVEMENT ######
# # Number of trees in random forest
# # Number of trees is not a parameter that should be tuned, but just set large enough usually. There is no risk of overfitting in random forest with growing number of # trees, as they are trained independently from each other. 
# n_estimators = list(range(200, 3000, 200))

# # Number of features to consider at every split
# max_features = ['auto', 'sqrt', None, 'log2']

# # Maximum number of levels in tree
# # Max depth is a parameter that most of the times should be set as high as possible, but possibly better performance can be achieved by setting it lower.
# max_depth = list(range(10, 200, 10))
# max_depth.append(None)

# # Minimum number of samples required to split a node
# # Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can also lead to # under-fitting hence depending on the level of underfitting or overfitting, you can tune the values for min_samples_split.
# min_samples_split = [2, 5, 10, 20, 30, 40]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 7, 12, 14, 16 ,20]

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'bootstrap': bootstrap
# }

# print(random_grid)

# # New Random Forest Classifier to house optimal parameters
# rf = RandomForestClassifier()

# # Specfiy the details of our Randomized Search
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model
# rf_random.fit(X_train, y_train)


# # With the new Random Classifier trained we can proceed to our regular steps, prediction.
# rf_random.predict(X_test)


# '''
#     ACCURACY
# '''
# # Once the predictions have been made, then grab the accuracy score.
# print('Correct Prediction (%): ', accuracy_score(y_test, rf_random.predict(X_test), normalize = True) * 100.0)


# '''
#     CLASSIFICATION REPORT
# '''
# # Define the traget names
# target_names = ['Down Day', 'Up Day']

# # Build a classifcation report
# report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

# # Add it to a data frame, transpose it for readability.
# report_df = pd.DataFrame(report).transpose()
# print(report_df)
# print('\n')

# '''
#     FEATURE IMPORTANCE
# '''
# # Calculate feature importance and store in pandas series
# feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# print(feature_imp)

# print(rf_random.best_estimator_)