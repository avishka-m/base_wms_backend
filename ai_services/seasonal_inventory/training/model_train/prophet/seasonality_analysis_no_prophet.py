import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import warnings
warnings.filterwarnings('ignore')

def comprehensive_seasonality_detection_no_prophet(df, products_per_category=3):
    """
    Comprehensive seasonality detection WITHOUT using Prophet
    Uses statistical methods, decomposition, and spectral analysis across all categories
    """
    print("="*80)
    print("COMPREHENSIVE SEASONALITY DETECTION (NO PROPHET REQUIRED)")
    print("="*80)
    
    categories = df['category'].unique()
    category_seasonality_results = {}
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"ANALYZING CATEGORY: {category.upper()}")
        print(f"{'='*60}")
        
        # Get top products from this category
        category_df = df[df['category'] == category]
        product_counts = category_df['product_id'].value_counts()
        top_products = product_counts.head(products_per_category).index.tolist()
        
        print(f"Analyzing {len(top_products)} representative products:")
        for i, product in enumerate(top_products, 1):
            count = product_counts[product]
            print(f"  {i}. {product}: {count} data points")
        
        category_results = {}
        
        for product in top_products:
            print(f"\n--- PRODUCT: {product} ---")
            
            # Filter data for this product
            product_data = df[df['product_id'] == product][['ds', 'y']].copy()
            product_data = product_data.sort_values('ds').reset_index(drop=True)
            product_data['ds'] = pd.to_datetime(product_data['ds'])
            
            if len(product_data) < 365:
                print(f"⚠️  Insufficient data ({len(product_data)} days)")
                continue
            
            # Add time features
            product_data['month'] = product_data['ds'].dt.month
            product_data['quarter'] = product_data['ds'].dt.quarter
            product_data['day_of_week'] = product_data['ds'].dt.dayofweek
            product_data['week_of_year'] = product_data['ds'].dt.isocalendar().week
            
            analysis_results = {}
            
            # ==================================================
            # 1. MONTHLY/YEARLY SEASONALITY ANALYSIS
            # ==================================================
            print("Monthly Seasonality Analysis:")
            
            # Calculate monthly averages
            monthly_stats = product_data.groupby('month')['y'].agg(['mean', 'std', 'count'])
            monthly_cv = monthly_stats['std'] / monthly_stats['mean']
            overall_monthly_cv = monthly_cv.mean()
            
            # Find peak and trough months
            peak_month = monthly_stats['mean'].idxmax()
            trough_month = monthly_stats['mean'].idxmin()
            peak_trough_ratio = monthly_stats['mean'].max() / monthly_stats['mean'].min()
            
            # Statistical test for monthly differences (ANOVA)
            month_groups = [product_data[product_data['month'] == m]['y'].values 
                           for m in range(1, 13) if len(product_data[product_data['month'] == m]) > 0]
            
            if len(month_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*month_groups)
                monthly_significant = p_value < 0.05
            else:
                f_stat, p_value = 0, 1.0
                monthly_significant = False
            
            # Kruskal-Wallis test (non-parametric)
            if len(month_groups) >= 3:
                kw_stat, kw_p = stats.kruskal(*month_groups)
                monthly_significant_np = kw_p < 0.05
            else:
                kw_stat, kw_p = 0, 1.0
                monthly_significant_np = False
            
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            print(f"  Peak month: {month_names[peak_month]} (avg: {monthly_stats['mean'][peak_month]:.2f})")
            print(f"  Trough month: {month_names[trough_month]} (avg: {monthly_stats['mean'][trough_month]:.2f})")
            print(f"  Peak/Trough ratio: {peak_trough_ratio:.2f}")
            print(f"  Monthly CV: {overall_monthly_cv:.3f}")
            print(f"  ANOVA p-value: {p_value:.4f} ({'SIGNIFICANT' if monthly_significant else 'NOT SIGNIFICANT'})")
            print(f"  Kruskal-Wallis p-value: {kw_p:.4f} ({'SIGNIFICANT' if monthly_significant_np else 'NOT SIGNIFICANT'})")
            
            analysis_results['monthly'] = {
                'peak_month': peak_month,
                'peak_month_name': month_names[peak_month],
                'trough_month': trough_month,
                'trough_month_name': month_names[trough_month],
                'peak_trough_ratio': peak_trough_ratio,
                'monthly_cv': overall_monthly_cv,
                'anova_significant': monthly_significant,
                'anova_p_value': p_value,
                'kruskal_significant': monthly_significant_np,
                'kruskal_p_value': kw_p,
                'monthly_averages': monthly_stats['mean'].to_dict()
            }
            
            # ==================================================
            # 2. WEEKLY SEASONALITY ANALYSIS
            # ==================================================
            print("\nWeekly Seasonality Analysis:")
            
            # Calculate weekly averages
            weekly_stats = product_data.groupby('day_of_week')['y'].agg(['mean', 'std', 'count'])
            weekly_cv = weekly_stats['std'] / weekly_stats['mean']
            overall_weekly_cv = weekly_cv.mean()
            
            # Find peak and trough days
            peak_day = weekly_stats['mean'].idxmax()
            trough_day = weekly_stats['mean'].idxmin()
            
            # Statistical test for weekly differences
            dow_groups = [product_data[product_data['day_of_week'] == d]['y'].values 
                         for d in range(7) if len(product_data[product_data['day_of_week'] == d]) > 0]
            
            if len(dow_groups) >= 3:
                f_stat_w, p_value_w = stats.f_oneway(*dow_groups)
                weekly_significant = p_value_w < 0.05
                kw_stat_w, kw_p_w = stats.kruskal(*dow_groups)
                weekly_significant_np = kw_p_w < 0.05
            else:
                f_stat_w, p_value_w = 0, 1.0
                kw_stat_w, kw_p_w = 0, 1.0
                weekly_significant = False
                weekly_significant_np = False
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            print(f"  Peak day: {day_names[peak_day]} (avg: {weekly_stats['mean'][peak_day]:.2f})")
            print(f"  Trough day: {day_names[trough_day]} (avg: {weekly_stats['mean'][trough_day]:.2f})")
            print(f"  Weekly CV: {overall_weekly_cv:.3f}")
            print(f"  ANOVA p-value: {p_value_w:.4f} ({'SIGNIFICANT' if weekly_significant else 'NOT SIGNIFICANT'})")
            print(f"  Kruskal-Wallis p-value: {kw_p_w:.4f} ({'SIGNIFICANT' if weekly_significant_np else 'NOT SIGNIFICANT'})")
            
            analysis_results['weekly'] = {
                'peak_day': peak_day,
                'peak_day_name': day_names[peak_day],
                'trough_day': trough_day,
                'trough_day_name': day_names[trough_day],
                'weekly_cv': overall_weekly_cv,
                'anova_significant': weekly_significant,
                'anova_p_value': p_value_w,
                'kruskal_significant': weekly_significant_np,
                'kruskal_p_value': kw_p_w,
                'weekly_averages': weekly_stats['mean'].to_dict()
            }
            
            # ==================================================
            # 3. SPECTRAL ANALYSIS (FREQUENCY DOMAIN)
            # ==================================================
            print("\nSpectral Analysis:")
            
            if len(product_data) >= 730:  # Need at least 2 years for reliable frequency analysis
                y_values = product_data['y'].values
                
                # Remove trend (first difference)
                y_detrended = np.diff(y_values)
                
                # Compute periodogram
                freqs, power = periodogram(y_detrended, fs=1.0)  # fs=1.0 for daily data
                
                # Convert frequencies to periods (in days)
                periods = 1.0 / freqs[1:]  # Skip the DC component
                power = power[1:]
                
                # Find dominant periods
                # Look for periods between 2 and 400 days
                valid_mask = (periods >= 2) & (periods <= 400)
                valid_periods = periods[valid_mask]
                valid_power = power[valid_mask]
                
                if len(valid_periods) > 0:
                    # Find top 5 periods
                    top_indices = np.argsort(valid_power)[-5:][::-1]
                    dominant_periods = valid_periods[top_indices]
                    dominant_powers = valid_power[top_indices]
                    
                    print(f"  Top dominant periods (days):")
                    for i, (period, pow_val) in enumerate(zip(dominant_periods, dominant_powers)):
                        if period >= 300:
                            period_type = f"~{period/365.25:.1f} years"
                        elif period >= 25:
                            period_type = f"~{period/30.44:.1f} months"
                        elif period >= 6:
                            period_type = f"~{period/7:.1f} weeks"
                        else:
                            period_type = f"{period:.1f} days"
                        print(f"    {i+1}. {period:.1f} days ({period_type}) - Power: {pow_val:.2e}")
                    
                    analysis_results['spectral'] = {
                        'dominant_periods': dominant_periods.tolist(),
                        'dominant_powers': dominant_powers.tolist()
                    }
                else:
                    print(f"  No significant periods found")
                    analysis_results['spectral'] = {'dominant_periods': [], 'dominant_powers': []}
            else:
                print(f"  Insufficient data for spectral analysis")
                analysis_results['spectral'] = {'dominant_periods': [], 'dominant_powers': []}
            
            # ==================================================
            # 4. OVERALL SEASONALITY SCORE
            # ==================================================
            seasonality_score = 0
            seasonality_evidence = []
            
            # Monthly seasonality evidence
            if analysis_results['monthly']['anova_significant']:
                seasonality_score += 30
                seasonality_evidence.append("Monthly ANOVA significant")
            if analysis_results['monthly']['peak_trough_ratio'] > 1.5:
                seasonality_score += 20
                seasonality_evidence.append(f"Strong monthly variation (ratio: {analysis_results['monthly']['peak_trough_ratio']:.2f})")
            
            # Weekly seasonality evidence
            if analysis_results['weekly']['anova_significant']:
                seasonality_score += 25
                seasonality_evidence.append("Weekly ANOVA significant")
            
            # Spectral evidence
            spectral_periods = analysis_results['spectral']['dominant_periods']
            for period in spectral_periods:
                if 350 <= period <= 380:  # Yearly cycle
                    seasonality_score += 15
                    seasonality_evidence.append("Annual cycle in spectrum")
                    break
            for period in spectral_periods:
                if 6 <= period <= 8:  # Weekly cycle
                    seasonality_score += 10
                    seasonality_evidence.append("Weekly cycle in spectrum")
                    break
            
            seasonality_score = min(seasonality_score, 100)  # Cap at 100
            
            print(f"\nOverall Seasonality Assessment:")
            print(f"  Seasonality Score: {seasonality_score}/100")
            print(f"  Evidence: {', '.join(seasonality_evidence) if seasonality_evidence else 'No strong seasonal evidence'}")
            
            if seasonality_score >= 70:
                seasonality_strength = "STRONG"
            elif seasonality_score >= 40:
                seasonality_strength = "MODERATE"
            elif seasonality_score >= 20:
                seasonality_strength = "WEAK"
            else:
                seasonality_strength = "MINIMAL"
            
            print(f"  Seasonality Strength: {seasonality_strength}")
            
            analysis_results['overall'] = {
                'seasonality_score': seasonality_score,
                'seasonality_strength': seasonality_strength,
                'evidence': seasonality_evidence
            }
            
            category_results[product] = analysis_results
        
        category_seasonality_results[category] = category_results
    
    return category_seasonality_results

def compare_seasonality_across_categories_no_prophet(category_results):
    """
    Compare seasonality patterns across categories without Prophet
    """
    print("\n" + "="*80)
    print("CROSS-CATEGORY SEASONALITY COMPARISON")
    print("="*80)
    
    # Analyze patterns by category
    category_summary = {}
    
    for category, products in category_results.items():
        print(f"\n{'='*50}")
        print(f"CATEGORY SUMMARY: {category.upper()}")
        print(f"{'='*50}")
        
        if not products:
            print("No products analyzed for this category")
            continue
        
        # Aggregate results for this category
        monthly_peaks = []
        weekly_peaks = []
        seasonality_scores = []
        peak_trough_ratios = []
        
        products_with_monthly_seasonality = 0
        products_with_weekly_seasonality = 0
        total_products = len(products)
        
        for product, analysis in products.items():
            if 'monthly' in analysis:
                monthly_peaks.append(analysis['monthly']['peak_month'])
                peak_trough_ratios.append(analysis['monthly']['peak_trough_ratio'])
                if analysis['monthly']['anova_significant']:
                    products_with_monthly_seasonality += 1
            
            if 'weekly' in analysis:
                weekly_peaks.append(analysis['weekly']['peak_day'])
                if analysis['weekly']['anova_significant']:
                    products_with_weekly_seasonality += 1
            
            if 'overall' in analysis:
                seasonality_scores.append(analysis['overall']['seasonality_score'])
        
        # Calculate category statistics
        avg_seasonality_score = np.mean(seasonality_scores) if seasonality_scores else 0
        avg_peak_trough_ratio = np.mean(peak_trough_ratios) if peak_trough_ratios else 0
        
        monthly_seasonality_pct = (products_with_monthly_seasonality / total_products) * 100
        weekly_seasonality_pct = (products_with_weekly_seasonality / total_products) * 100
        
        print(f"Products analyzed: {total_products}")
        print(f"Average seasonality score: {avg_seasonality_score:.1f}/100")
        print(f"Average peak/trough ratio: {avg_peak_trough_ratio:.2f}")
        print(f"Products with monthly seasonality: {products_with_monthly_seasonality}/{total_products} ({monthly_seasonality_pct:.1f}%)")
        print(f"Products with weekly seasonality: {products_with_weekly_seasonality}/{total_products} ({weekly_seasonality_pct:.1f}%)")
        
        # Most common peak months/days
        if monthly_peaks:
            from collections import Counter
            month_counter = Counter(monthly_peaks)
            most_common_month = month_counter.most_common(1)[0]
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            print(f"Most common peak month: {month_names[most_common_month[0]]} ({most_common_month[1]}/{len(monthly_peaks)} products)")
        
        if weekly_peaks:
            from collections import Counter
            day_counter = Counter(weekly_peaks)
            most_common_day = day_counter.most_common(1)[0]
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            print(f"Most common peak day: {day_names[most_common_day[0]]} ({most_common_day[1]}/{len(weekly_peaks)} products)")
        
        # Determine seasonality strength for category
        if avg_seasonality_score >= 70:
            category_seasonality_strength = "STRONG"
        elif avg_seasonality_score >= 40:
            category_seasonality_strength = "MODERATE"
        elif avg_seasonality_score >= 20:
            category_seasonality_strength = "WEAK"
        else:
            category_seasonality_strength = "MINIMAL"
        
        print(f"Category seasonality strength: {category_seasonality_strength}")
        
        category_summary[category] = {
            'total_products': total_products,
            'avg_seasonality_score': avg_seasonality_score,
            'seasonality_strength': category_seasonality_strength,
            'monthly_seasonality_pct': monthly_seasonality_pct,
            'weekly_seasonality_pct': weekly_seasonality_pct,
            'avg_peak_trough_ratio': avg_peak_trough_ratio,
            'monthly_peaks': monthly_peaks,
            'weekly_peaks': weekly_peaks
        }
    
    return category_summary

def recommend_modeling_approach_no_prophet(category_summary):
    """
    Recommend modeling approach based on seasonality analysis (without Prophet bias)
    """
    print("\n" + "="*80)
    print("MODELING RECOMMENDATIONS BASED ON SEASONALITY ANALYSIS")
    print("="*80)
    
    # Overall analysis
    total_categories = len(category_summary)
    categories_with_strong_seasonality = sum(1 for cat in category_summary.values() 
                                           if cat['seasonality_strength'] in ['STRONG', 'MODERATE'])
    
    avg_seasonality_score = np.mean([cat['avg_seasonality_score'] for cat in category_summary.values()])
    
    print(f"📊 OVERALL SEASONALITY ASSESSMENT:")
    print(f"  Categories analyzed: {total_categories}")
    print(f"  Categories with significant seasonality: {categories_with_strong_seasonality}/{total_categories}")
    print(f"  Average seasonality score: {avg_seasonality_score:.1f}/100")
    
    # Check consistency across categories
    peak_months = []
    peak_days = []
    for category, summary in category_summary.items():
        if summary['monthly_peaks']:
            from collections import Counter
            most_common = Counter(summary['monthly_peaks']).most_common(1)[0][0]
            peak_months.append(most_common)
        if summary['weekly_peaks']:
            from collections import Counter
            most_common = Counter(summary['weekly_peaks']).most_common(1)[0][0]
            peak_days.append(most_common)
    
    unique_peak_months = len(set(peak_months)) if peak_months else 0
    unique_peak_days = len(set(peak_days)) if peak_days else 0
    
    print(f"\n🔍 SEASONALITY PATTERN CONSISTENCY:")
    print(f"  Unique peak months across categories: {unique_peak_months}")
    print(f"  Unique peak days across categories: {unique_peak_days}")
    
    # Model recommendations
    print(f"\n🎯 RECOMMENDED MODELING APPROACHES:")
    
    if avg_seasonality_score >= 60:
        print(f"\n✅ STRONG SEASONALITY DETECTED - TIME SERIES MODELS RECOMMENDED")
        
        if unique_peak_months <= 1 and unique_peak_days <= 1:
            print(f"\n🎯 OPTION 1: UNIVERSAL SEASONAL MODEL (RECOMMENDED)")
            print(f"   → All categories show similar seasonal patterns")
            print(f"   → Use 1 model for all products with strong seasonality")
            print(f"   → Best models: SARIMA, Prophet, Seasonal Decomposition + ML")
        elif unique_peak_months <= 2:
            print(f"\n🎯 OPTION 1: CATEGORY-BASED MODELS (RECOMMENDED)")
            print(f"   → {total_categories} models (one per category)")
            print(f"   → Each category has consistent internal seasonality")
            print(f"   → Best models: Prophet, SARIMA, Seasonal Decomposition + ML")
        else:
            print(f"\n🎯 OPTION 1: HYBRID APPROACH (RECOMMENDED)")
            print(f"   → Category models for main trends")
            print(f"   → Individual scaling factors for products")
            print(f"   → Best models: Prophet with custom seasonalities, Ensemble methods")
        
        print(f"\n   📚 SPECIFIC MODEL RECOMMENDATIONS:")
        print(f"   🥇 PROPHET: Best for automatic seasonality detection")
        print(f"      ✅ Handles yearly and weekly seasonality automatically")
        print(f"      ✅ Robust to missing data and outliers")
        print(f"      ✅ Provides uncertainty intervals")
        print(f"      ✅ Easy to interpret and tune")
        
        print(f"\n   🥈 SARIMA: Best for classical time series approach")
        print(f"      ✅ Excellent for strong, regular seasonality")
        print(f"      ✅ Well-established statistical foundation")
        print(f"      ❌ Requires more manual tuning")
        
        print(f"\n   🥉 SEASONAL DECOMPOSITION + ML: Best for complex patterns")
        print(f"      ✅ Separates trend, seasonality, and residuals")
        print(f"      ✅ Can use any ML model on deseasonalized data")
        print(f"      ✅ Very flexible approach")
    
    elif avg_seasonality_score >= 30:
        print(f"\n⚠️  MODERATE SEASONALITY - HYBRID APPROACHES RECOMMENDED")
        print(f"\n🎯 OPTION 1: ENSEMBLE METHODS")
        print(f"   → Combine seasonal and non-seasonal models")
        print(f"   → Weight based on seasonal strength per product")
        print(f"   → Models: Prophet + XGBoost, SARIMA + Random Forest")
        
        print(f"\n🎯 OPTION 2: FEATURE-RICH ML MODELS")
        print(f"   → Add time-based features (month, day of week, etc.)")
        print(f"   → Use XGBoost, Random Forest, or Neural Networks")
        print(f"   → Let the model learn seasonal patterns from features")
    
    else:
        print(f"\n❌ WEAK/MINIMAL SEASONALITY - NON-SEASONAL MODELS RECOMMENDED")
        print(f"\n🎯 RECOMMENDED APPROACHES:")
        print(f"   📈 TREND-BASED: Linear/polynomial regression with time trends")
        print(f"   🤖 ML-BASED: XGBoost, Random Forest with lag features")
        print(f"   📊 MOVING AVERAGES: Simple moving average or exponential smoothing")
        print(f"   ⚠️  AVOID: SARIMA, Prophet (seasonality components not needed)")
    
    # Implementation strategy
    print(f"\n🚀 IMPLEMENTATION STRATEGY FOR 2000 PRODUCTS:")
    
    if categories_with_strong_seasonality >= total_categories * 0.8:
        print(f"\n📋 STEP 1: Start with category-level models")
        print(f"   → Build {total_categories} seasonal models (one per category)")
        print(f"   → Use Prophet or SARIMA for each category")
        print(f"   → Scale predictions to individual products")
        
        print(f"\n📋 STEP 2: Identify products needing individual models")
        print(f"   → Products with very different patterns from category average")
        print(f"   → High-volume products where accuracy is critical")
        print(f"   → Build individual models only for these products")
        
        print(f"\n📋 STEP 3: Performance monitoring")
        print(f"   → Track accuracy by product and category")
        print(f"   → Switch to individual models if category model underperforms")
    else:
        print(f"\n📋 STEP 1: Segment products by seasonality strength")
        print(f"   → Strong seasonal: Use seasonal models (Prophet, SARIMA)")
        print(f"   → Weak seasonal: Use ML models (XGBoost, Random Forest)")
        print(f"   → No seasonal: Use simple trend/moving average models")
        
        print(f"\n📋 STEP 2: Start simple, then optimize")
        print(f"   → Begin with category-level models")
        print(f"   → Add individual models for top products")
        print(f"   → Use ensemble methods for best performance")
    
    return category_summary

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    print(f"Data shape: {df.shape}")
    print(f"Categories: {df['category'].unique()}")
    
    # Run comprehensive seasonality analysis
    category_results = comprehensive_seasonality_detection_no_prophet(df, products_per_category=3)
    
    # Compare across categories
    category_summary = compare_seasonality_across_categories_no_prophet(category_results)
    
    # Get recommendations
    recommend_modeling_approach_no_prophet(category_summary)
