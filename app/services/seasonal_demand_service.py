from datetime import datetime, timedelta
from app.utils.database import get_collection

def update_seasonal_scores():
    demand_forecast_col = get_collection("demand_forecast")
    seasonal_demand_col = get_collection("seasonal_demand")
    today = datetime.utcnow()
    start_date = today - timedelta(days=365)

    # Map months to seasons 
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    categories = demand_forecast_col.distinct("category")
    for category in categories:
        forecasts = list(demand_forecast_col.find({
            "category": category,
            "date": {"$gte": start_date, "$lte": today}
        }))
        season_scores = {}
        for forecast in forecasts:
            date = forecast["date"]
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            season = get_season(date)
            season_scores.setdefault(season, []).append(forecast["yhat"])
        for season, yhat_list in season_scores.items():
            avg_score = sum(yhat_list) / len(yhat_list) if yhat_list else 0
            seasonal_demand_col.update_one(
                {"category": category, "season": season},
                {"$set": {
                    "seasonal_score": avg_score,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
if __name__ == "__main__":
    update_seasonal_scores()
    print("Seasonal scores updated.")
