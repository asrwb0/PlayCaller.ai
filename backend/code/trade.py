# import all necessary modules
import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional, cast, Any
warnings.filterwarnings("ignore")

POSITIONS = ["QB", "RB", "WR", "TE"]

# positions like QB and TE are scarcer
POSITION_SCARCITY = {"QB": 1.15, "TE": 1.10, "WR": 1.00, "RB": 0.95}
INJURY_PENALTY = {"Low": 0.0, "Moderate": 3.0, "High": 10.0, "Critical": 20.0}

# define thresholds for trade decisions based on the net value for you
VERDICT_THRESHOLDS = [
    ("STRONG ACCEPT", 15.0),
    ("ACCEPT", 5.0),
    ("FAIR", -5.0),
    ("DECLINE", -15.0),
    ("STRONG DECLINE", -1000), # create a very low threshold for super bad trade offers
]

class PlayerProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pos_baseline = (df.groupby(["position", "season"])["fantasy_points"].mean().to_dict())

    def _baseline(self, position: str, season: int) -> float:
        return self.pos_baseline.get((position, season), 10.0)

    def get_profile(self, player_name: str) -> dict | None:
        matching_rows = []
        for i, row in self.df.iterrows():
            name = row["player_display_name"]
            if pd.notna(name) and player_name.lower() in name.lower():
                matching_rows.append(row)

        rows = pd.DataFrame(matching_rows).copy()
        
        if len(rows) == 0:
            # split a player's name and take just their last name because users often type things like
            # "Mahomes" instead of "Patrick Mahomes"
            last_name = player_name.split()[-1]
            
            matching_rows = []
            for i, row in self.df.iterrows():
                name = row["player_display_name"]
                if pd.notna(name) and last_name.lower() in name.lower():
                    matching_rows.append(row)
            
            rows = pd.DataFrame(matching_rows).copy()
            if rows.empty:
                return None

        rows = rows.sort_values(["season", "week"], ascending = False)
        pos = rows["position"].iloc[0]
        season = rows["season"].iloc[0]
        name = rows["player_display_name"].iloc[0]
        team = rows["recent_team"].iloc[0]

        last_3_weeks = rows.head(3)
        last_5_weeks = rows.head(5)
        # the last season played is 16 games
        last_season = rows.head(16)
        season_rows = rows[rows["season"] == season]

        # these metrics are the players most recent averages, total points for the season and calculated projection
        last_3_weeks_avg    = last_3_weeks["fantasy_points"].mean()
        last_5_weeks_avg    = last_5_weeks["fantasy_points"].mean()
        total_season_points   = season_rows["fantasy_points"].mean()
        projected = (last_3_weeks_avg * 0.50) + (last_5_weeks_avg * 0.30) + (total_season_points * 0.20)

        pts_std = last_season["fantasy_points"].std()
        if np.isnan(pts_std):
            pts_std = 8.0

        # this determines if a player's snap share/usage is going up or down
        last_3_weeks_snap_share_avg = last_3_weeks["snap_share"].mean()
        last_8_weeks_snap_share_avg = last_season.head(8)["snap_share"].mean()
        usage_trend = last_3_weeks_snap_share_avg - last_8_weeks_snap_share_avg

        pts_above_avg = projected - self._baseline(pos, season)
        total_games = len(rows)
        boom_rate  = (last_season["fantasy_points"] >= 20).mean()
        bust_rate = (last_season["fantasy_points"] >= 8).mean()

        # 30 fpts is a very good game so we can normalize the data around that by dividing by 30
        # this gives a "proportion" of how good they did
        pts_score = projected / 30.0
        if pts_score > 1.0:
            pts_score = 1.0
        pts_score *= 40 # this is 40 percent of the overall score
        
        # this essentially calculates how consistent a player is by comparing their performance
        # to the standard deviation of points (the lower the standard deviation, the closer their scores to the mean)
        # this means they are more consistent
        consistency_score = 1.0 - (pts_std / 15.0)
        if consistency_score < 0.0:
            consistency_score = 0.0
        consistency_score *= 20
        
        usage_score = usage_trend + 0.5
        if usage_score < 0.0:
            usage_score = 0.0
        if usage_score > 1.0:
            usage_score = 1.0
        usage_score *= 15

        performance_score = pts_above_avg + 5.0
        if performance_score < 0.0:
            performance_score = 0.0
        performance_score = min(performance_score / 20.0, 1.0)
        performance_score *= 15

        longevity_score = total_games / 80.0
        if longevity_score > 1.0:
            longevity_score = 1.0
        longevity_score *= 10

        total_score = pts_score + consistency_score + usage_score + performance_score + longevity_score

        # position scarcity is based on the values we defined all the way at the top
        # this is used because a TE averaging 20 is MUCH more valuable than a WR averaging 20
        position_multiplier = POSITION_SCARCITY.get(pos, 1.0)
        trade_value = total_score * position_multiplier
        trade_value = round(min(trade_value, 100.0), 1)

        return {
            "player": name,
            "position": pos,
            "team": team,
            "projected_pts": round(projected, 1),
            "pts_std": round(pts_std, 1),
            "pts_above_avg": round(pts_above_avg, 1),
            "usage_trend": round(usage_trend, 3),
            "snap_share_recent": round(float(last_3_weeks_snap_share_avg), 2),
            "boom_rate": round(float(boom_rate), 2),
            "bust_rate": round(float(bust_rate), 2),
            "total_games": total_games,
            "raw_trade_value": trade_value,
            "trade_value": trade_value,
            "injury_risk_tier": "Unknown",
            "injury_risk_percentage": "?",
            "injury_risk_color_code": "grey",
            "injury_penalty": 0.0,
            "injury_flags": [], }

    def suggest(self, name: str, n: int = 5) -> list:
        prefix = name.lower()[:4]
        all_matching_names = self.df["player_display_name"].unique()

        matching_names = []
        for player_name in all_matching_names:
            if prefix in player_name.lower():
                matching_names.append(player_name)

        # this is useful because some players have the same first name so allowing the user to choose between them is good
        return matching_names[:n]

class TradeEvaluator:
    def __init__(self):
        self.profiler: Optional[PlayerProfiler] = None
        self.fantasy_model: Optional[Any] = None  # PointsPredictor imported at runtime
        self.injury_model: Optional[Any] = None  # InjuryRiskPredictor imported at runtime
        self._ready = False

    def load(self, model_dir: str = "./models", data_path: str = "./code/final_weekly_stats.csv"):
        player_data = pd.read_csv(data_path)
        # only choose players from the right position
        player_data = player_data[player_data["position"].isin(POSITIONS)].copy()
        player_data = player_data.sort_values(["player_display_name", "season", "week"])

        self.profiler = PlayerProfiler(player_data)

        from points import PointsPredictor
        self.fantasy_model = PointsPredictor()
        if self.fantasy_model is not None:
            self.fantasy_model.load_model(model_dir)

        from injury import InjuryRiskPredictor
        self.injury_model = InjuryRiskPredictor()
        if self.injury_model is not None:
            self.injury_model.load_model(model_dir)

        self._ready = True

    def _score_player(self, player_name: str) -> dict | None:
        if self.profiler is None or self.fantasy_model is None or self.injury_model is None:
            raise ValueError("Models not loaded. Call load() first.")
        
        profiler = cast(PlayerProfiler, self.profiler)
        fantasy_model = cast(Any, self.fantasy_model)
        injury_model = cast(Any, self.injury_model)
        
        profile = profiler.get_profile(player_name)
        if profile is None:
            return None

        # a player's profile shouldn't be entirely the model's prediction, because some players
        # might just be in a slump so their past average/performance should play a role
        ml_prediction = fantasy_model.predict_for_player(player_name)
        if ml_prediction and "predicted_points" in ml_prediction:
            blended_points = (ml_prediction["predicted_points"] * 0.60 + profile["projected_pts"] * 0.40)
            profile["projected_pts"] = round(blended_points, 1)

            # calculate points above the baseline for the player's position
            position_baseline = profiler._baseline(profile["position"], 2024)
            points_above_baseline = profile["projected_pts"] - position_baseline
            profile["pts_above_avg"] = round(points_above_baseline, 1)

            # same normalization as before
            scaled_projected = profile["projected_pts"] / 30.0
            capped_scaled_projected = min(scaled_projected, 1.0)
            projected_points_score = capped_scaled_projected * 40

            adjusted_points_above_baseline = points_above_baseline + 5.0
            non_negative_adjusted = max(adjusted_points_above_baseline, 0.0)
            scaled_performance = non_negative_adjusted / 20.0
            capped_performance = min(scaled_performance, 1.0)
            performance_score = capped_performance * 15

            consistency_ratio = 1.0 - (profile["pts_std"] / 15.0)
            non_negative_consistency = max(consistency_ratio, 0.0)
            consistency_score = non_negative_consistency * 20

            shifted_usage_trend = profile["usage_trend"] + 0.5
            usage_non_negative = max(shifted_usage_trend, 0.0)
            usage_capped = min(usage_non_negative, 1.0)
            usage_score = usage_capped * 15

            scaled_longevity = profile["total_games"] / 80.0
            capped_longevity = min(scaled_longevity, 1.0)
            longevity_score = capped_longevity * 10
    
            total_raw_score = projected_points_score + performance_score + consistency_score + usage_score + longevity_score

            # adjust for positions because some positions, like tight end, average less points
            scarcity_multiplier = POSITION_SCARCITY.get(profile["position"], 1.0)
            profile["raw_trade_value"] = min(round(total_raw_score * scarcity_multiplier, 1), 100.0)
            profile["trade_value"] = profile["raw_trade_value"]

        # if a player gets injured a lot, their value should decrease because missing games means no points
        injury_summary = injury_model.predict_for_player(player_name)
        risk_tier = injury_summary.get("risk_tier", "Unknown")
        injury_penalty = INJURY_PENALTY.get(risk_tier, 0.0)
        profile["injury_risk_tier"] = risk_tier
        profile["injury_penalty"] = injury_penalty
        profile["injury_risk_pct"] = injury_summary.get("risk_pct", "?")
        profile["injury_risk_color"] = injury_summary.get("risk_color", "grey")
        profile["injury_flags"] = injury_summary.get("key_flags", [])

        # subtract injury penalty from trade value
        profile["trade_value"] = max(round(profile["raw_trade_value"] - injury_penalty, 1), 0.0)

        return profile

    def evaluate(self, giving: list, receiving: list) -> dict:
        # since some players rookie years happened after the data set ends (2024),
        # a list of players that weren't found in the database is created
        not_found = []
        giving_players, receiving_players = [], []

        for name in giving:
            curr_score = self._score_player(name)
            giving_players.append(curr_score)

        for name in receiving:
            curr_score = self._score_player(name)
            receiving_players.append(curr_score)

        giving_total = 0
        for name in giving_players:
            giving_total += name["trade_value"]
        giving_val = round(giving_total, 1)
        
        receiving_total = 0
        for name in receiving_players:
            receiving_total += name["trade_value"]
        receiving_val = round(receiving_total, 1)

        giving_pts_total = 0
        for name in giving_players:
            giving_pts_total += name["projected_pts"]
        giving_pts = round(giving_pts_total, 1)

        receiving_pts_total = 0
        for name in receiving_players:
            receiving_pts_total += name["projected_pts"]
        receiving_pts = round(receiving_pts_total, 1)

        # this is used to make the final verdict and calculates how much you stand to gain/lose from the trade
        net_swing = round(receiving_val - giving_val, 1)

        verdict = "STRONG DECLINE"
        for label, threshold in VERDICT_THRESHOLDS:
            if net_swing >= threshold:
                verdict = label
                break

        # this adds warnings for player injuries and usage, which is useful because some players have
        # newly aggravated injuries or team situations
        warnings = []
        all_players = giving_players + receiving_players

        for player in all_players:
            player_name = player.get("player", "Unknown")
            injury_tier = player.get("injury_risk_tier", "Unknown")
            injury_pct  = player.get("injury_risk_percentage", "?")
            injury_flags = player.get("injury_flags", [])
            usage_trend = player.get("usage_trend", 0.0)

            if injury_tier == "High" or injury_tier == "Critical":
                if len(injury_flags) > 0:
                    first_flag = injury_flags[0]
                else:
                    first_flag = ""
                warning_text = player_name + " — " + injury_tier + " injury risk (" + str(injury_pct) + "): " + first_flag
                warnings.append(warning_text)

            if usage_trend < -0.12:
                usage_warning = player_name + " — snap share declining (" + "{0:+.0%}".format(usage_trend) + " trend last 3 vs last 8 games)"
                warnings.append(usage_warning)

        summary = _build_summary(verdict, net_swing, giving_players, receiving_players, giving_pts, receiving_pts, )

        trade_result = {}
        trade_result["verdict"] = verdict
        trade_result["net_value_swing"] = net_swing
        trade_result["giving_total"] = giving_val
        trade_result["receiving_total"] = receiving_val
        trade_result["giving_pts_total"] = giving_pts
        trade_result["receiving_pts_total"] = receiving_pts
        trade_result["giving_players"] = giving_players
        trade_result["receiving_players"] = receiving_players
        trade_result["summary"] = summary
        trade_result["warnings"] = warnings
        trade_result["not_found"] = not_found

        return trade_result

def _build_summary(verdict, net_swing, giving, receiving, g_pts, r_pts) -> str:
    for g_player in giving:
        g_names  = " + ".join(g_player["player"])
    for r_player in receiving:
        r_names  = " + ".join(r_player["player"])
    
    pts_diff = round(r_pts - g_pts, 1)

    # this creates a string that is used in the trade summary that says how many points are gained/lost from the trade
    if pts_diff >= 0:
        pts_str = f"gaining {pts_diff:+.1f} projected pts/wk"
    else:
        pts_str = f"losing {abs(pts_diff):.1f} projected pts/wk"

    phrases = {
        "STRONG ACCEPT": f"Definitely take it. Trading {g_names} for {r_names} "
                          f"nets you +{net_swing:.1f} in trade value while {pts_str}.",
        "ACCEPT":         f"Slight benefit for you. Giving {g_names} for {r_names} "
                          f"gives you +{net_swing:.1f} in trade value while {pts_str}.",
        "FAIR":           f"About even for both teams. {g_names} for {r_names} is within "
                          f"{abs(net_swing):.1f} points of fair value ({pts_str}).",
        "DECLINE":        f"Slight loss for you. {g_names} for {r_names} costs you "
                          f"{abs(net_swing):.1f} in trade value while {pts_str}.",
        "STRONG DECLINE": f"Do NOT accept this trade at all. Giving {g_names} for {r_names} "
                          f"costs you {abs(net_swing):.1f} in trade value while {pts_str}.", }

    base = phrases.get(verdict, f"Net swing: {net_swing:+.1f}.")

    notes = []

    # snap share is the best way to describe a player's usage and is used to gauge injury risk
    for p in receiving:
        if p.get("injury_risk_tier") in ("High", "Critical"):
            notes.append(f"{p['player']} carries {p['injury_risk_tier'].lower()} "
                         f"injury risk ({p.get('injury_risk_percentage','?')})")
        if p.get("usage_trend", 0) > 0.10:
            notes.append(f"{p['player']} has rising usage ({p['usage_trend']:+.0%})")
    for p in giving:
        if p.get("usage_trend", 0) < -0.10:
            notes.append(f"{p['player']} has declining usage ({p['usage_trend']:+.0%})")

    if notes:
        base += " Note: " + "; ".join(notes) + "."
    return base

# this is used to funnel data into the FastAPI API layer because we only want one instance per player
_singleton = None

def evaluate_trade(giving: list, receiving: list, data_path: str = "./final_weekly_stats.csv", model_dir:  str = "./models") -> dict:
    global _singleton
    # if there isn't an instance of the trade evaluator class yet, we should make one
    if _singleton is None:
        t = TradeEvaluator()
        t.load(model_dir, data_path)
        _singleton = t
    return _singleton.evaluate(giving, receiving)


def main():
    # creates an instance of the class
    evaluator = TradeEvaluator()
    evaluator.load(model_dir = "./models", data_path = "./final_weekly_stats.csv",)

    # these are some scenarios that might happen in a league
    scenarios = [
        {
            # this trade is lopsided because Justin Jefferson on his own is a better asset than Tyreek Hill
            # so getting Justin Jefferson AND another player is a must-take offer in any league
            "label": "Completely lopsided trade",
            "giving": ["Tyreek Hill"],
            "receiving": ["Justin Jefferson", "Tony Pollard"],
        },
        {
            # this trade is somewhat even because Travis Kelce is getting older and Mark Andrews is more consistent
            "label": "A somewhat fair TE swap",
            "giving": ["Travis Kelce"],
            "receiving": ["Mark Andrews"],
        },
        {
            # Christian McCaffrey has had high injury risk for his entire career because he touches the ball so much
            # getting CeeDee, who is much less injury prone, and Derrick Henry who is a solid RB is a pretty good deal
            "label": "Selling high on an injury-prone player",
            "giving": ["Christian McCaffrey"],
            "receiving": ["Derrick Henry", "CeeDee Lamb"],
        },
        {
            # getting Josh Allen for Mahomes is an upgrade because Allen has more rushing upside
            # Allen also goes off for 40 point games sometimes because his team needs him more
            "label": "Decent QB upgrade",
            "giving": ["Patrick Mahomes"],
            "receiving": ["Josh Allen"],
        },
        {
            # this is a horrible trade for the exact same reasons as the first sample trade 
            "label": "Horrible trade",
            "giving": ["Justin Jefferson", "Derrick Henry"],
            "receiving": ["Tyreek Hill"],
        },
    ]

    for s in scenarios:
        print(f"{s['label']}")
        result = evaluator.evaluate(s["giving"], s["receiving"])
        if result:
            print(f"  Verdict: {result.get('verdict', 'UNKNOWN')}")
            print(f"  Net swing: {result.get('net_swing', 0):.1f} points")
            print(f"  Summary: {result.get('summary', '')}")

if __name__ == "__main__":
    main()