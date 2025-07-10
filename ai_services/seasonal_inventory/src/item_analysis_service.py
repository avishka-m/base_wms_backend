"""
Item Analysis Service for Seasonal Inventory Prediction

This service provides detailed item-to-item analysis and recommendations
using Prophet forecasting models.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

from .models.prophet_forecaster import ProphetForecaster
from .data_collection.wms_data_extractor import WMSDataExtractor
from ..config import PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItemAnalysisService:
    """
    Service for analyzing individual items and their demand patterns.
    """
    
    def __init__(self):
        """Initialize the item analysis service."""
        self.forecaster = ProphetForecaster()
        self.wms_extractor = WMSDataExtractor()
        self.cache = {}
        
        logger.info("🔍 Item Analysis Service initialized")
    
    async def analyze_item(self, product_id: str, retrain: bool = False) -> Dict:
        """
        Comprehensive analysis of a specific item.
        
        Args:
            product_id: Product identifier
            retrain: Whether to retrain the model with latest data
            
        Returns:
            Complete item analysis
        """
        logger.info(f"🔍 Analyzing item: {product_id}")
        
        try:
            # Get or train model for this product
            if retrain or product_id not in self.forecaster.models:
                await self._train_product_model(product_id)
            
            # Get detailed analysis
            analysis = self.forecaster.get_item_details(product_id)
            
            if "error" in analysis:
                return analysis
            
            # Add real-time data if available
            current_stock = await self._get_current_stock(product_id)
            if current_stock:
                analysis["current_stock_info"] = current_stock
            
            # Add supplier information
            supplier_info = await self._get_supplier_info(product_id)
            if supplier_info:
                analysis["supplier_info"] = supplier_info
            
            # Generate inventory recommendations
            inventory_recommendations = self._generate_inventory_recommendations(analysis)
            analysis["inventory_recommendations"] = inventory_recommendations
            
            # Cache the analysis
            self.cache[product_id] = analysis
            
            logger.info(f"✅ Analysis complete for {product_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {product_id}: {e}")
            return {"error": str(e)}
    
    async def _train_product_model(self, product_id: str) -> bool:
        """Train a model for specific product using WMS data."""
        try:
            # Extract historical data for this product
            historical_data = await self.wms_extractor.extract_sales_data()
            
            if historical_data.empty:
                logger.warning(f"⚠️ No historical data found for {product_id}")
                return False
            
            # Filter for specific product
            product_data = historical_data[historical_data['product_id'] == product_id]
            
            if len(product_data) < 14:  # Need at least 2 weeks
                logger.warning(f"⚠️ Insufficient data for {product_id}: {len(product_data)} records")
                return False
            
            # Train the model
            result = self.forecaster.train_model(product_data, product_id)
            
            if result.get("success"):
                logger.info(f"✅ Model trained for {product_id}")
                return True
            else:
                logger.error(f"❌ Training failed for {product_id}: {result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error training model for {product_id}: {e}")
            return False
    
    async def _get_current_stock(self, product_id: str) -> Optional[Dict]:
        """Get current stock information for a product."""
        try:
            # This would query your current inventory levels
            # For now, return mock data structure
            return {
                "current_stock": 0,  # Would be fetched from database
                "reserved_stock": 0,
                "available_stock": 0,
                "reorder_point": 0,
                "max_stock": 0,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not get current stock for {product_id}: {e}")
            return None
    
    async def _get_supplier_info(self, product_id: str) -> Optional[Dict]:
        """Get supplier information for a product."""
        try:
            # This would query supplier data from your database
            return {
                "supplier_id": "SUP-001",  # Would be fetched from database
                "lead_time_days": 7,
                "min_order_qty": 50,
                "unit_cost": 0.0,
                "last_delivery": None
            }
        except Exception as e:
            logger.warning(f"⚠️ Could not get supplier info for {product_id}: {e}")
            return None
    
    def _generate_inventory_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate specific inventory management recommendations."""
        recommendations = []
        
        forecast_insights = analysis.get("forecast_insights", {})
        historical_stats = analysis.get("historical_stats", {})
        
        # Reorder point recommendation
        avg_daily_demand = forecast_insights.get("average_daily_demand", 0)
        lead_time = 7  # Default lead time
        safety_stock = avg_daily_demand * 3  # 3 days safety stock
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        
        recommendations.append({
            "type": "reorder_point",
            "title": "Recommended Reorder Point",
            "value": int(reorder_point),
            "description": f"Reorder when stock drops to {int(reorder_point)} units",
            "calculation": f"({avg_daily_demand:.1f} daily demand × {lead_time} lead days) + {safety_stock:.1f} safety stock"
        })
        
        # Order quantity recommendation
        predicted_monthly_demand = forecast_insights.get("predicted_total_demand", 0)
        if predicted_monthly_demand > 0:
            monthly_order_qty = predicted_monthly_demand * 1.1  # 10% buffer
            
            recommendations.append({
                "type": "order_quantity",
                "title": "Recommended Monthly Order Quantity",
                "value": int(monthly_order_qty),
                "description": f"Order {int(monthly_order_qty)} units for next month",
                "calculation": f"{predicted_monthly_demand:.1f} predicted demand + 10% buffer"
            })
        
        # Peak demand preparation
        peak_demand = forecast_insights.get("peak_demand_value", 0)
        peak_date = forecast_insights.get("peak_demand_date", "")
        
        if peak_demand > avg_daily_demand * 1.5:
            recommendations.append({
                "type": "peak_preparation",
                "title": "Peak Demand Alert",
                "value": int(peak_demand),
                "description": f"Prepare for high demand ({int(peak_demand)} units) around {peak_date[:10]}",
                "urgency": "high"
            })
        
        # Trend-based recommendations
        trend = forecast_insights.get("demand_trend", "")
        if trend == "increasing":
            recommendations.append({
                "type": "trend_alert",
                "title": "Increasing Demand Trend",
                "description": "Consider increasing stock levels due to growing demand",
                "action": "increase_inventory"
            })
        elif trend == "decreasing":
            recommendations.append({
                "type": "trend_alert", 
                "title": "Decreasing Demand Trend",
                "description": "Consider promotional activities or reduced ordering",
                "action": "promotional_campaign"
            })
        
        return recommendations
    
    async def compare_similar_items(self, product_id: str, category: str = None) -> Dict:
        """
        Compare an item with similar products in the same category.
        
        Args:
            product_id: Target product ID
            category: Product category (if known)
            
        Returns:
            Comparison analysis
        """
        logger.info(f"🔄 Finding similar items to {product_id}")
        
        try:
            # Get all products in the same category or similar patterns
            similar_products = await self._find_similar_products(product_id, category)
            
            if not similar_products:
                return {"error": "No similar products found"}
            
            # Add the target product to comparison
            if product_id not in similar_products:
                similar_products.append(product_id)
            
            # Get comparison analysis
            comparison = self.forecaster.compare_products(similar_products)
            
            if "error" in comparison:
                return comparison
            
            # Add insights specific to the target product
            target_analysis = await self.analyze_item(product_id)
            comparison["target_product"] = {
                "product_id": product_id,
                "analysis": target_analysis
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error comparing similar items: {e}")
            return {"error": str(e)}
    
    async def _find_similar_products(self, product_id: str, category: str = None) -> List[str]:
        """Find products similar to the given product."""
        try:
            # This would implement similarity logic based on:
            # - Same category
            # - Similar demand patterns  
            # - Same supplier
            # - Price range
            
            # For now, return mock similar products
            return [f"SKU-{i:03d}" for i in range(1, 6) if f"SKU-{i:03d}" != product_id]
            
        except Exception as e:
            logger.warning(f"⚠️ Could not find similar products: {e}")
            return []
    
    async def batch_analyze_products(self, product_ids: List[str]) -> Dict:
        """
        Analyze multiple products in batch.
        
        Args:
            product_ids: List of product IDs to analyze
            
        Returns:
            Batch analysis results
        """
        logger.info(f"🔄 Batch analyzing {len(product_ids)} products")
        
        results = {}
        failed = []
        
        for product_id in product_ids:
            try:
                analysis = await self.analyze_item(product_id)
                if "error" not in analysis:
                    results[product_id] = analysis
                else:
                    failed.append({"product_id": product_id, "error": analysis["error"]})
            except Exception as e:
                failed.append({"product_id": product_id, "error": str(e)})
        
        return {
            "batch_analysis_date": datetime.now().isoformat(),
            "total_requested": len(product_ids),
            "successful": len(results),
            "failed": len(failed),
            "results": results,
            "failed_items": failed,
            "summary": self._create_batch_summary(results)
        }
    
    def _create_batch_summary(self, results: Dict) -> Dict:
        """Create summary statistics for batch analysis."""
        if not results:
            return {}
        
        total_demand = sum(
            r["forecast_insights"]["predicted_total_demand"] 
            for r in results.values() 
            if "forecast_insights" in r
        )
        
        avg_volatility = sum(
            r["historical_stats"]["demand_volatility"] 
            for r in results.values() 
            if "historical_stats" in r
        ) / len(results)
        
        return {
            "total_predicted_demand": total_demand,
            "average_volatility": avg_volatility,
            "products_with_increasing_trend": len([
                r for r in results.values() 
                if r.get("forecast_insights", {}).get("demand_trend") == "increasing"
            ]),
            "products_with_decreasing_trend": len([
                r for r in results.values() 
                if r.get("forecast_insights", {}).get("demand_trend") == "decreasing"
            ]),
            "high_volatility_products": [
                product_id for product_id, analysis in results.items()
                if analysis.get("historical_stats", {}).get("demand_volatility", 0) > avg_volatility * 1.5
            ]
        }
    
    async def get_category_insights(self, category: str) -> Dict:
        """
        Get insights for all products in a category.
        
        Args:
            category: Product category
            
        Returns:
            Category-level insights
        """
        logger.info(f"📊 Analyzing category: {category}")
        
        try:
            # Get all products in category (would query database)
            products_in_category = await self._get_products_by_category(category)
            
            if not products_in_category:
                return {"error": f"No products found in category: {category}"}
            
            # Batch analyze all products in category
            batch_results = await self.batch_analyze_products(products_in_category)
            
            # Add category-specific insights
            category_insights = {
                "category": category,
                "analysis_date": datetime.now().isoformat(),
                "total_products": len(products_in_category),
                "batch_results": batch_results,
                "category_trends": self._analyze_category_trends(batch_results),
                "top_performers": self._identify_top_performers(batch_results),
                "recommendations": self._generate_category_recommendations(batch_results)
            }
            
            return category_insights
            
        except Exception as e:
            logger.error(f"❌ Error analyzing category {category}: {e}")
            return {"error": str(e)}
    
    async def _get_products_by_category(self, category: str) -> List[str]:
        """Get all product IDs in a category."""
        try:
            # This would query your database for products in the category
            # For now, return mock data
            return [f"{category}-{i:03d}" for i in range(1, 11)]
        except Exception as e:
            logger.warning(f"⚠️ Could not get products for category {category}: {e}")
            return []
    
    def _analyze_category_trends(self, batch_results: Dict) -> Dict:
        """Analyze trends across the category."""
        results = batch_results.get("results", {})
        
        if not results:
            return {}
        
        trends = {"increasing": 0, "decreasing": 0, "stable": 0}
        
        for analysis in results.values():
            trend = analysis.get("forecast_insights", {}).get("demand_trend", "stable")
            trends[trend] = trends.get(trend, 0) + 1
        
        return {
            "trend_distribution": trends,
            "dominant_trend": max(trends, key=trends.get),
            "trend_consistency": max(trends.values()) / len(results) * 100
        }
    
    def _identify_top_performers(self, batch_results: Dict) -> Dict:
        """Identify top performing products in the category."""
        results = batch_results.get("results", {})
        
        if not results:
            return {}
        
        # Sort by predicted demand
        sorted_by_demand = sorted(
            results.items(),
            key=lambda x: x[1].get("forecast_insights", {}).get("predicted_total_demand", 0),
            reverse=True
        )
        
        return {
            "highest_demand": [item[0] for item in sorted_by_demand[:5]],
            "lowest_volatility": [],  # Would implement based on volatility
            "most_consistent": [],    # Would implement based on consistency metrics
        }
    
    def _generate_category_recommendations(self, batch_results: Dict) -> List[str]:
        """Generate recommendations for the entire category."""
        recommendations = []
        
        summary = batch_results.get("summary", {})
        
        if summary.get("products_with_increasing_trend", 0) > summary.get("products_with_decreasing_trend", 0):
            recommendations.append("📈 Category shows overall growth trend - consider expanding inventory")
        
        if summary.get("high_volatility_products"):
            recommendations.append("⚠️ Some products show high volatility - implement safety stock strategies")
        
        return recommendations if recommendations else ["No specific category recommendations"]
    
    async def close(self):
        """Clean up resources."""
        await self.wms_extractor.close()


async def main():
    """Demo the item analysis service."""
    print("🔍 Item Analysis Service Demo")
    print("=" * 50)
    
    service = ItemAnalysisService()
    
    try:
        print("\n🔄 Service capabilities:")
        print("• analyze_item(product_id) - Complete item analysis")
        print("• compare_similar_items(product_id) - Find and compare similar products")
        print("• batch_analyze_products(product_ids) - Analyze multiple products")
        print("• get_category_insights(category) - Category-level analysis")
        
        print("\n💡 Example item analysis structure:")
        print("""
{
    "product_id": "SKU-001",
    "historical_stats": {
        "average_daily_demand": 45.2,
        "demand_volatility": 12.3,
        "total_demand": 1356
    },
    "forecast_insights": {
        "predicted_total_demand": 1200,
        "demand_trend": "increasing",
        "peak_demand_date": "2025-07-15"
    },
    "inventory_recommendations": [
        {
            "type": "reorder_point",
            "title": "Recommended Reorder Point",
            "value": 150,
            "description": "Reorder when stock drops to 150 units"
        }
    ],
    "recommendations": [
        "📈 Increasing demand trend. Plan for inventory buildup."
    ]
}
""")
        
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
