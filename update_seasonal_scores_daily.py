# update_seasonal_scores_daily.py
"""
Standalone script to update seasonal scores in the seasonal_demand collection.
Can be scheduled to run daily using Windows Task Scheduler.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ai_services.seasonal_inventory.seasonal_demand_service import update_seasonal_scores

if __name__ == "__main__":
    update_seasonal_scores()
    print("Seasonal scores updated successfully.")
