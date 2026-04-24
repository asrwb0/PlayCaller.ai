import pandas as pd
import numpy as np
from typing import Optional, List, Dict, cast
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class PointsPredictor:
    def __init__(self):
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_cols: Optional[List[str]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.player_lookup: Optional[pd.DataFrame] = None
        self.training_data: Optional[pd.DataFrame] = None

    def load_and_prepare_data(self, filepath = './code/final_weekly_stats.csv'):
        df = pd.read_csv(filepath)

        # only keep skill positions
        df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

        # make sure the target isn't a NaN
        df = df[df['fantasy_points'].notna()].copy()

        df = df[df['fantasy_points'] > 0].copy()

        # create a lookup for each player
        self.player_lookup = (df.sort_values(['season', 'week']).groupby('player_display_name').last().reset_index()[['player_display_name', 'position', 'recent_team']])
        self.training_data = df.copy()

        return df

    def engineer_features(self, df):

        rolling_features = [
            'fantasy_points_avg_3w',
            'fantasy_points_avg_5w',
            'targets_avg_3w',
            'targets_avg_5w',
            'carries_avg_3w',
            'carries_avg_5w',
            'receiving_yards_avg_3w',
            'receiving_yards_avg_5w',
        ]

        lag_features = [
            'fantasy_points_lag_1',
            'fantasy_points_lag_2',
            'targets_lag_1',
            'targets_lag_2',
            'carries_lag_1',
            'carries_lag_2',
        ]

        usage_features = [
            'target_share',
            'carry_share',
            'touch_share',
            'snap_share',
        ]

        basic_features = [
            'targets',
            'carries',
            'receptions',
            'offensive_snaps',
        ]

        categorical_features = ['position']
        self.feature_cols = []

        for feat_list in [rolling_features, lag_features, usage_features, basic_features]:
            for feat in feat_list:
                if feat in df.columns:
                    self.feature_cols.append(feat)

        for cat in categorical_features:
            if cat in df.columns:
                le = LabelEncoder()
                df[f'{cat}_encoded'] = le.fit_transform(df[cat].astype(str))
                self.label_encoders[cat] = le
                self.feature_cols.append(f'{cat}_encoded')

        return df

    def split(self, df):

        df = df.sort_values(['season', 'week'])
        last_season = df['season'].max()

        train_df = df[df['season'] < last_season].copy()
        test_df  = df[df['season'] == last_season].copy()

        def make_X_y(frame):
            X = frame[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
            y = frame['fantasy_points']
            return X, y

        X_train, y_train = make_X_y(train_df)
        X_test,  y_test  = make_X_y(test_df)

        return X_train, X_test, y_train, y_test, test_df

    def train(self, X_train, y_train, X_test, y_test):
        self.model = GradientBoostingRegressor(
            max_depth = 4, 
            min_samples_split = 10, 
            min_samples_leaf = 5, 
            learning_rate = 0.05, 
            n_estimators = 500, 
            subsample = 0.8,
            max_features = 0.8, 
            loss = 'huber', 
            random_state = 42, 
            verbose = 0, 
            validation_fraction = 0.15, 
            n_iter_no_change = 50, 
            tol = 0.001
        )
        self.model.fit(X_train, y_train)

        return self.model

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained")
        y_tr_pred  = self.model.predict(X_train)
        y_te_pred  = self.model.predict(X_test)

        tr_mae  = mean_absolute_error(y_train, y_tr_pred)
        te_mae  = mean_absolute_error(y_test,  y_te_pred)
        tr_rmse = np.sqrt(mean_squared_error(y_train, y_tr_pred))
        te_rmse = np.sqrt(mean_squared_error(y_test,  y_te_pred))
        tr_r2   = r2_score(y_train, y_tr_pred)
        te_r2   = r2_score(y_test,  y_te_pred)

        gap_pts = te_mae - tr_mae
        gap_pct = (gap_pts / tr_mae) * 100

        errors = np.abs(y_test.values - y_te_pred)
        w2 = (errors <= 2).mean() * 100
        w3 = (errors <= 3).mean() * 100
        w5 = (errors <= 5).mean() * 100

        return dict(train_mae = tr_mae, test_mae = te_mae, train_rmse = tr_rmse, test_rmse = te_rmse, train_r2 = tr_r2, 
                    test_r2 = te_r2, mae_gap_pct = gap_pct, within_2_pct = w2, within_3_pct = w3, y_test_pred = y_te_pred, y_test = y_test,)

    def cross_validate(self, df):
        X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['fantasy_points'].values

        tscv = TimeSeriesSplit(n_splits = 5)
        cv_scores = []

        for i, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            m = GradientBoostingRegressor(max_depth = 4, learning_rate = 0.05, n_estimators = 200, subsample = 0.8, random_state = 42,)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            mae = mean_absolute_error(y[val_idx], m.predict(X.iloc[val_idx]))
            cv_scores.append(mae)

        return cv_scores

    def predict_for_player(self, player_name, season = None, week = None):
        if self.training_data is None or self.feature_cols is None or self.model is None:
            raise ValueError("Model or training data not loaded")
       
        training_data = cast(pd.DataFrame, self.training_data)
        feature_cols = cast(List[str], self.feature_cols)
        mask = training_data['player_display_name'].str.contains(player_name, case = False, na = False)
        player_rows = training_data[mask].copy()

        if player_rows.empty:
            # match only by last name
            last_name = player_name.split()[-1]
            mask2 = training_data['player_display_name'].str.contains(last_name, case = False, na = False)
            player_rows = training_data[mask2].copy()

        if player_rows.empty:
            # look for similar names
            all_names = training_data['player_display_name'].unique()
            suggestions = [n for n in all_names if player_name.lower()[:4] in n.lower()][:5]
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions)}")
            return None

        if season is not None:
            player_rows = player_rows[player_rows['season'] == season]
        if week is not None:
            player_rows = player_rows[player_rows['week'] < week]  # only past weeks

        if player_rows.empty:
            print(f"  No data for {player_name} in the timeframe.")
            return None

        player_rows = player_rows.sort_values(['season', 'week'], ascending = False)
        recent_games = player_rows.head(5)

        feature_vector = {}
        for feat in feature_cols:
            if feat in recent_games.columns:
                val = recent_games[feat].mean()
                if np.isnan(val):
                    val = player_rows[feat].mean()
                if np.isnan(val):
                    val = 0.0
                feature_vector[feat] = val
            else:
                feature_vector[feat] = 0.0

        X_pred = pd.DataFrame([feature_vector])[feature_cols]
        X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

        model = cast(GradientBoostingRegressor, self.model)
        prediction = float(model.predict(X_pred)[0])
        prediction = max(0.0, min(prediction, 65.0))

        actual_pts = recent_games['fantasy_points'].dropna()
        if len(actual_pts) >= 2:
            std_dev = float(actual_pts.std())
        else:
            pos = recent_games['position'].iloc[0]
            pos_std = {'QB': 7.5, 'RB': 6.5, 'WR': 6.5, 'TE': 5.0}
            std_dev = pos_std.get(pos, 6.5)

        margin = round(1.28 * std_dev, 1)
        low    = round(max(0.0, prediction - margin), 1)
        high   = round(prediction + margin, 1)

        player_info = recent_games.iloc[0]
        name   = player_info.get('player_display_name', player_name)
        pos    = player_info.get('position', '?')
        team   = player_info.get('recent_team', '?')
        recent_avg = round(float(actual_pts.mean()), 1) if len(actual_pts) else '?'

        return {
            'player':            name,
            'position':          pos,
            'team':              team,
            'predicted_points':  round(prediction, 1),
            'low':               low,
            'high':              high,
            'range':             f"{low}–{high}",
            'recent_avg':        recent_avg,
        }

    def save_model(self, output_dir = './models'):
        import os
        os.makedirs(output_dir, exist_ok = True)

        joblib.dump(self.model,          f'{output_dir}/xgboost_model.pkl')
        joblib.dump(self.feature_cols,   f'{output_dir}/feature_columns.pkl')
        joblib.dump(self.label_encoders, f'{output_dir}/label_encoders.pkl')
        joblib.dump(self.player_lookup,  f'{output_dir}/player_lookup.pkl')
        joblib.dump(self.training_data,  f'{output_dir}/training_data.pkl')

    def load_model(self, model_dir = './models'):
        self.model          = joblib.load(f'{model_dir}/xgboost_model.pkl')
        self.feature_cols   = joblib.load(f'{model_dir}/feature_columns.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        self.player_lookup  = joblib.load(f'{model_dir}/player_lookup.pkl')
        self.training_data  = joblib.load(f'{model_dir}/training_data.pkl')

_predictor_singleton = None

def predict_fantasy_points(player_name, season = None, week = None, data_path = './final_weekly_stats.csv', model_dir = './models'):
    global _predictor_singleton
    import os

    if _predictor_singleton is None:
        p = PointsPredictor()

        try:
            p.load_model(model_dir)
        except FileNotFoundError:
            df = p.load_and_prepare_data(data_path)
            df = p.engineer_features(df)
            X_train, X_test, y_train, y_test, _ = p.split(df)
            p.train(X_train, y_train, X_test, y_test)
            p.evaluate_model(X_train, y_train, X_test, y_test)
            p.save_model(model_dir)

        _predictor_singleton = p

    return _predictor_singleton.predict_for_player(player_name, season = season, week = week)