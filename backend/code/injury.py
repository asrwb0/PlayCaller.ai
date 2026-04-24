import pandas as pd
import numpy as np
from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, classification_report, )
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

POSITIONS = ["QB", "RB", "WR", "TE"]

RISK_TIERS = [
    (0.00, 0.15, "Low", "green"),
    (0.15, 0.28, "Moderate", "yellow"),
    (0.28, 0.42, "High", "orange"),
    (0.42, 1.01, "Critical", "red"),
]

FEATURE_COLS = [
    "snap_share", "touch_share", "carry_share", "target_share",
    "touches", "carries", "targets", "offensive_snaps",
    "rush_tkl_loss", "rush_btkl", "fum",
    "snap_share_lag1", "snap_share_lag2", "snap_trend", "snap_share_avg3",
    "touches_lag1", "touches_lag2", "touches_avg3",
    "carries_lag1", "rush_tkl_loss_3w", "fum_3w",
    "fantasy_points_avg_3w", "fantasy_points_avg_5w",
    "fantasy_points_lag_1", "fantasy_points_lag_2",
    "carries_avg_3w", "carries_avg_5w",
    "targets_avg_3w", "targets_avg_5w",
    "season_carries", "season_touches", "season_snaps",
    "week", "position_enc",
]

def build_injury_label(df):
    df = df.sort_values(["player_display_name", "season", "week"]).reset_index(drop = True)
    grp = df.groupby("player_display_name")

    df["next_snap_share"] = grp["snap_share"].shift(-1)
    df["next_week"] = grp["week"].shift(-1)
    df["next_season"] = grp["season"].shift(-1)
    df["week_gap"] = df["next_week"] - df["week"]

    missed_game = (df["next_season"] == df["season"]) & (df["week_gap"] > 1)
    snapped_out = (
        (df["next_season"] == df["season"]) &
        (df["next_snap_share"] < 0.15) &
        (df["snap_share"] > 0.40)
    )
    df["injured_next_game"] = (missed_game | snapped_out).astype(int)
    df = df[df["next_season"] == df["season"]].copy()
    return df


def engineer_injury_features(df):
    df = df.sort_values(["player_display_name", "season", "week"]).reset_index(drop = True)
    grp = df.groupby(["player_display_name", 'season'])

    df["snap_share_lag1"] = grp["snap_share"].shift(1)
    df["snap_share_lag2"] = grp["snap_share"].shift(2)
    df["snap_trend"]      = df["snap_share"] - df["snap_share_lag1"]
    df["snap_share_avg3"] = grp["snap_share"].transform(
        lambda x: x.rolling(3, min_periods = 1).mean().shift(1))

    df["touches_lag1"] = grp["touches"].shift(1)
    df["touches_lag2"] = grp["touches"].shift(2)
    df["touches_avg3"] = grp["touches"].transform(
        lambda x: x.rolling(3, min_periods = 1).mean().shift(1))
    df["carries_lag1"] = grp["carries"].shift(1)

    df["rush_tkl_loss_3w"] = grp["rush_tkl_loss"].transform(
        lambda x: x.rolling(3, min_periods = 1).sum().shift(1))
    df["fum_3w"] = grp["fum"].transform(
        lambda x: x.rolling(3, min_periods = 1).sum().shift(1))

    df["season_carries"] = (grp["carries"].cumsum() - df["carries"]).clip(lower = 0)
    df["season_touches"] = (grp["touches"].cumsum() - df["touches"]).clip(lower = 0)
    df["season_snaps"] = (grp["offensive_snaps"].cumsum() - df["offensive_snaps"]).clip(lower = 0)

    le = LabelEncoder()
    df["position_enc"] = le.fit_transform(df["position"].astype(str))
    return df, le

class InjuryRiskPredictor:
    def __init__(self):
        self.gb_model: Optional[GradientBoostingClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.feature_cols  = FEATURE_COLS

    def load_and_prepare(self, filepath = './data/data.csv'):
        df = pd.read_csv(filepath)
        df = df[df["position"].isin(POSITIONS)].copy()
        df = build_injury_label(df)
        rate = df["injured_next_game"].mean()
        return df

    def engineer_features(self, df):
        df, le = engineer_injury_features(df)
        self.label_encoder = le
        return df

    def split(self, df):
        df_clean    = df.dropna(subset = ["injured_next_game"]).copy()
        last_season = df_clean["season"].max()
        train_mask  = df_clean["season"] < last_season

        X = df_clean[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = df_clean["injured_next_game"]
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]

        self.training_data = df_clean.copy()
        return X_train, X_test, y_train, y_test, df_clean

    def train(self, X_train, y_train, X_test, y_test):
        cw = compute_class_weight("balanced", classes = np.array([0,1]), y = y_train.values)

        self.gb_model = GradientBoostingClassifier(n_estimators = 300, max_depth = 4, learning_rate = 0.05, subsample = 0.8, min_samples_leaf = 10, random_state = 42, validation_fraction = 0.15, n_iter_no_change = 30, tol = 1e-4,)
        self.gb_model.fit(X_train, y_train)

        self.rf_model = RandomForestClassifier(n_estimators = 300, max_depth = 6, min_samples_leaf = 10, class_weight = "balanced", random_state = 42, n_jobs = -1,)
        self.rf_model.fit(X_train, y_train)
        return self

    def evaluate(self, X_train, y_train, X_test, y_test):
        if self.gb_model is None or self.rf_model is None:
            raise ValueError("Models not trained")
        gb_tr  = self.gb_model.predict_proba(X_train)[:,1]
        gb_te  = self.gb_model.predict_proba(X_test)[:,1]
        rf_te  = self.rf_model.predict_proba(X_test)[:,1]
        ens_te = (gb_te + rf_te) / 2

        ens_auc = roc_auc_score(y_test, ens_te)
        ens_ap  = average_precision_score(y_test, ens_te)
        ens_bs  = brier_score_loss(y_test, ens_te)

        gap = roc_auc_score(y_train, gb_tr) - roc_auc_score(y_test, gb_te)

        y_pred = (ens_te >= 0.28).astype(int)

        if self.gb_model is None:
            raise ValueError("GB model not trained")
        fi = pd.Series(self.gb_model.feature_importances_, index = self.feature_cols).sort_values(ascending = False)

        return {"ensemble_auc": ens_auc, "ensemble_ap": ens_ap,
                "ensemble_brier": ens_bs, "overfit_gap": gap}

    
    def cross_validate(self, df_clean):
        X = df_clean[self.feature_cols].fillna(0).replace([np.inf,-np.inf],0)
        y = df_clean["injured_next_game"].values
        aucs = []
        for i, (tr, val) in enumerate(TimeSeriesSplit(n_splits = 5).split(X)):
            m = GradientBoostingClassifier(n_estimators = 150, max_depth = 4, learning_rate = 0.05, subsample = 0.8, random_state = 42)
            m.fit(X.iloc[tr], y[tr])
            auc = roc_auc_score(y[val], m.predict_proba(X.iloc[val])[:,1])
            aucs.append(auc)
        return aucs

    def save_model(self, output_dir = "./models"):
        os.makedirs(output_dir, exist_ok = True)
        joblib.dump(self.gb_model,      f"{output_dir}/injury_gb_model.pkl")
        joblib.dump(self.rf_model,      f"{output_dir}/injury_rf_model.pkl")
        joblib.dump(self.label_encoder, f"{output_dir}/injury_label_encoder.pkl")
        joblib.dump(self.feature_cols,  f"{output_dir}/injury_feature_cols.pkl")
        joblib.dump(self.training_data, f"{output_dir}/injury_training_data.pkl")

    def load_model(self, model_dir = "./models"):
        self.gb_model      = joblib.load(f"{model_dir}/injury_gb_model.pkl")
        self.rf_model      = joblib.load(f"{model_dir}/injury_rf_model.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/injury_label_encoder.pkl")
        self.feature_cols  = joblib.load(f"{model_dir}/injury_feature_cols.pkl")
        self.training_data = joblib.load(f"{model_dir}/injury_training_data.pkl")

    def predict_for_player(self, player_name, season = None, week = None):
        if self.training_data is None or self.gb_model is None or self.rf_model is None:
            raise ValueError("Models or training data not loaded")

        # fuzzy name matching
        mask = self.training_data["player_display_name"].str.contains(player_name, case = False, na = False)
        rows = self.training_data[mask].copy()

        if rows.empty:
            last_name = player_name.split()[-1]
            rows = self.training_data[
                self.training_data["player_display_name"].str.contains(last_name, case = False, na = False)].copy()

        if rows.empty:
            all_names = self.training_data["player_display_name"].unique()
            sugg = [n for n in all_names if player_name.lower()[:4] in n.lower()][:5]
            return {"error": f"Player '{player_name}' not found.", "suggestions": sugg}

        if season is not None:
            rows = rows[rows["season"] == season]
        if week is not None:
            rows = rows[rows["week"] < week]
        if rows.empty:
            return {"error": f"No data for '{player_name}' in this timeframe."}

        rows   = rows.sort_values(["season", "week"], ascending = False)
        recent = rows.head(5)

        fv = {}
        for feat in self.feature_cols:
            val = recent[feat].mean() if feat in recent.columns else 0.0
            fv[feat] = 0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else val

        X_pred = pd.DataFrame([fv])[self.feature_cols].fillna(0)

        if self.gb_model is None or self.rf_model is None:
            raise ValueError("Models not trained")
        gb_p = self.gb_model.predict_proba(X_pred)[0, 1]
        rf_p = self.rf_model.predict_proba(X_pred)[0, 1]
        risk = float(max(0.0, min((gb_p + rf_p) / 2, 1.0)))

        tier = "Low"; color = "green"
        for lo, hi, label, col in RISK_TIERS:
            if lo <= risk < hi:
                tier = label; color = col; break

        flags = []
        snap_avg3  = fv.get("snap_share_avg3", 0)
        snap_now   = fv.get("snap_share", 0)
        if snap_avg3 > 0.3 and snap_now < snap_avg3 * 0.75:
            flags.append(f"Snap share dropped: {snap_avg3:.0%} avg -> {snap_now:.0%}")
        if fv.get("snap_trend", 0) < -0.15:
            flags.append(f"Declining snap share ({fv['snap_trend']:+.0%} last game)")
        if fv.get("season_carries", 0) > 200:
            flags.append(f"Heavy season carry load ({int(fv['season_carries'])} carries)")
        if fv.get("touches_avg3", 0) > 22:
            flags.append(f"High touch volume ({fv['touches_avg3']:.1f} avg last 3 games)")
        if fv.get("rush_tkl_loss_3w", 0) >= 4:
            flags.append(f"High contact: {int(fv['rush_tkl_loss_3w'])} TKL-losses last 3 wks")
        if fv.get("fum_3w", 0) >= 2:
            flags.append(f"Multiple fumbles recently ({int(fv['fum_3w'])} in last 3 games)")
        if fv.get("week", 0) >= 14:
            flags.append(f"Late-season fatigue (Week {int(fv['week'])})")
        if not flags:
            flags.append("No major workload concerns detected.")

        recs = {
            "Low":      "Safe to start. No significant injury concerns this week.",
            "Moderate": "Monitor practice reports mid-week before finalizing lineup.",
            "High":     "Check injury report Thursday/Friday. Have a backup ready.",
            "Critical": "High injury risk - strongly consider starting an alternative.",
        }

        info = rows.iloc[0]
        snap_vals = recent["snap_share"].dropna()

        return {
            "player":          str(info.get("player_display_name", player_name)),
            "position":        str(info.get("position", "?")),
            "team":            str(info.get("recent_team", "?")),
            "risk_score":      round(risk, 3),
            "risk_pct":        f"{risk:.0%}",
            "risk_tier":       tier,
            "risk_color":      color,
            "recommendation":  recs[tier],
            "key_flags":       flags,
            "recent_avg_snap": round(float(snap_vals.mean()), 3) if len(snap_vals) else None,
            "season_carries":  int(fv.get("season_carries", 0)),
            "season_touches":  int(fv.get("season_touches", 0)),
        }

_injury_singleton = None

def predict_injury_risk(player_name, season = None, week = None, data_path = "./final_weekly_stats.csv", model_dir = "./models"):
    global _injury_singleton
    if _injury_singleton is None:
        m = InjuryRiskPredictor()
        try:
            m.load_model(model_dir)
        except FileNotFoundError:
            df = m.load_and_prepare(data_path)
            df = m.engineer_features(df)
            X_tr, X_te, y_tr, y_te, df_c = m.split(df)
            m.train(X_tr, y_tr, X_te, y_te)
            m.save_model(model_dir)
        _injury_singleton = m
    return _injury_singleton.predict_for_player(player_name, season = season, week = week)