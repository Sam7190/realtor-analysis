import vars
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, DMatrix
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import warnings
warnings.simplefilter('ignore', category=pd.errors.SettingWithCopyWarning)

# This function will generate the training data and outcomes for a given zip code
def grab_training_records(final_df, zip, outcome_thresholds=vars.outcome_thresholds):
    postal_df = final_df[final_df['postal_code'] == zip].reset_index(0).drop(columns=['index'])
    training_sets = {i: [] for i in range(1, vars.max_months_predicted + 1)}
    outcome_sets = {i: [] for i in range(1, vars.max_months_predicted + 1)}
    appended = {i: 0 for i in range(1, vars.max_months_predicted + 1)}

    outcome_trend = postal_df[['month']].copy().reset_index(drop=True).iloc[:-1]

    price_after = postal_df['median_listing_price'].iloc[:-1].to_numpy()
    price_before = postal_df['median_listing_price'].iloc[1:].to_numpy()
    price_ratio = (price_after - price_before) / price_before
    outcome_trend['price_change'] = price_ratio

    outcome_trend['days_on_market'] = postal_df['median_days_on_market'].iloc[:-1].to_numpy()

    count_after = postal_df['page_view_count_per_property_vs_us'].iloc[:-1].to_numpy()
    count_before = postal_df['page_view_count_per_property_vs_us'].iloc[1:].to_numpy()
    count_ratio = (count_after - count_before) / count_before
    outcome_trend['view_count_change'] = count_ratio

    for i in range(len(postal_df)):
        for t in training_sets:
            j = i + t
            if j >= len(postal_df):
                continue

            # Get how many months the two records are separted by
            months_sep = postal_df.loc[i, 'month'] - postal_df.loc[j, 'month']
            months_int = months_sep.n
            if (months_int > vars.max_months_predicted):
                continue
            elif months_int < 1:
                raise AssertionError(f"Somehow months_int < 1 for {i}, {j}, {months_int}")

            # Calculate Price Outcomes
            price_before = postal_df.loc[i, 'median_listing_price']
            price_after = postal_df.loc[j, 'median_listing_price']
            price_ratio = (price_after - price_before) / price_before

            high_risk_price = int(price_ratio <= outcome_thresholds['high_risk']['price_change'])
            strong_market_price = int(price_ratio >= outcome_thresholds['strong_market']['price_change'])

            # Calculate Day on Market Outcomes
            dom_after = postal_df.loc[j, 'median_days_on_market']

            high_risk_dom = int(dom_after >= outcome_thresholds['high_risk']['days_on_market'])
            strong_market_dom = int(dom_after <= outcome_thresholds['strong_market']['days_on_market'])

            # Calculate View Count Outcomes
            count_before = postal_df.loc[i, 'page_view_count_per_property_vs_us']
            count_after = postal_df.loc[j, 'page_view_count_per_property_vs_us']
            count_ratio = (count_after - count_before) / count_before

            high_risk_count = int(count_ratio <= outcome_thresholds['high_risk']['view_count_change'])
            strong_market_count = int(count_ratio >= outcome_thresholds['strong_market']['view_count_change'])

            # Append to training and outcome sets
            training_sets[months_int].append(postal_df.loc[j, vars.features].to_numpy())
            outcome_sets[months_int].append([high_risk_price, strong_market_price,
                                             high_risk_dom, strong_market_dom,
                                             high_risk_count, strong_market_count])
            appended[months_int] += 1
    
    inference_point = np.array([postal_df.loc[0, vars.features].to_numpy()])
    for t in training_sets:
        training_sets[t] = np.array(training_sets[t])
        outcome_sets[t] = np.array(outcome_sets[t])

    return training_sets, outcome_sets, outcome_trend, inference_point


class XGBDegenerateClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_degenerate_ = None
        self.degenerate_class_ = None

    def fit_degen(self, X, y=None, **kwargs):
        # If all target values are the same, skip training
        if (sum(y) == len(y)) or (sum(y) == 0):
            self.is_degenerate_ = True
            self.degenerate_class_ = y[0]
        else:
            self.is_degenerate_ = False
            self.fit(X, y, **kwargs)
        return self

    def predict_degen(self, X):
        if self.is_degenerate_:
            return np.full((X.shape[0],), self.degenerate_class_)
        else:
            return self.predict(X)

    def predict_proba_degen(self, X):
        if self.is_degenerate_:
            proba = np.zeros((X.shape[0], 2))
            proba[:, self.degenerate_class_] = 1.0
            return proba
        else:
            return self.predict_proba(X)
        

def train_and_predict(training_sets, outcome_sets, inference_point):
    results, predictions = [], []
    #training_sets, outcome_sets, inference_point = grab_training_records(final_df, zip)

    for t in training_sets:
        print(f"\nTraining model for {t} months prediction with {len(training_sets[t])} records")
        
        predicted = [[] for _ in range(len(outcome_sets[t][0]))]
        for i in range(len(training_sets[t])):
            train_idx = np.ones((len(training_sets[t]),), dtype=bool)
            train_idx[i] = False
            X_train, X_val = training_sets[t][train_idx], training_sets[t][~train_idx]
            
            for outcome in range(len(outcome_sets[t][i])):
                y_train, y_val = outcome_sets[t][train_idx][:, outcome], outcome_sets[t][~train_idx][:, outcome]
                num_positives = np.sum(y_train)
                
                model = XGBDegenerateClassifier(eval_metric='logloss')
                print(f"  Outcome: {vars.outcomes[outcome]} | Records: {len(y_train)} | Targets: {num_positives}")
                model.fit_degen(X_train, y_train)
                preds = model.predict_proba_degen(X_val)
                predicted[outcome].append(preds[0, 1])

        predicted = np.array(predicted)

        R, P = [], []
        for outcome in range(len(outcome_sets[t][0])):
            y_true = outcome_sets[t][:, outcome]
            if np.all(y_true == 0) or np.all(y_true == 1):
                auc_score = None  # undefined
            else:
                auc_score = np.round(roc_auc_score(y_true, predicted[outcome]), 4)
            R.append(auc_score)

            # Final Prediction
            model = XGBDegenerateClassifier(eval_metric='logloss')
            model.fit_degen(training_sets[t], outcome_sets[t][:, outcome])
            
            final_pred = model.predict_proba_degen(inference_point)
            P.append(final_pred[0, 1])
            print(f"Months Out: {t} | Outcome {vars.outcomes[outcome]} | Prediction: {final_pred[0, 1]} | AUC: {auc_score}")
        results.append(R)
        predictions.append(P)
    
    return np.array(predictions), np.array(results)