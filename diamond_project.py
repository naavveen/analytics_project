import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   DIAMOND PRICE ANALYSIS + ML PREDICTION — NAVEEN")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
import seaborn as sns_data
df = sns_data.load_dataset('diamonds')

print(f"\n[ STEP 1 ] DATASET LOADED")
print(f"  Shape         : {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Price range   : ${df['price'].min():,} — ${df['price'].max():,}")
print(f"  Avg price     : ${df['price'].mean():,.0f}")
print(f"\n  Sample:")
print(df.head(3).to_string(index=False))

# ─────────────────────────────────────────
# 2. DATA CLEANING & VALIDATION
# ─────────────────────────────────────────
print(f"\n[ STEP 2 ] DATA CLEANING")

# Remove physically impossible dimensions (x, y, z = 0)
before = len(df)
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
print(f"  Removed {before - len(df)} rows with zero dimensions")

# Remove extreme outliers in y and z (data entry errors)
df = df[df['y'] < 20]
df = df[df['z'] < 10]
print(f"  Removed outlier dimension rows")
print(f"  Final shape: {df.shape}")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
df['volume']        = df['x'] * df['y'] * df['z']
df['price_per_carat'] = df['price'] / df['carat']

# Ordinal encoding for cut, color, clarity (they have natural order)
cut_order     = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_order   = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_order = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5,
                 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut_score']     = df['cut'].map(cut_order)
df['color_score']   = df['color'].map(color_order)
df['clarity_score'] = df['clarity'].map(clarity_order)

print(f"\n[ STEP 3 ] FEATURES ENGINEERED")
print(f"  volume          : x * y * z")
print(f"  price_per_carat : price / carat")
print(f"  cut/color/clarity → ordinal scores (quality ranking)")

# ─────────────────────────────────────────
# 4. EDA — KEY INSIGHTS
# ─────────────────────────────────────────
print(f"\n[ STEP 4 ] KEY INSIGHTS")

corr_carat  = df['carat'].corr(df['price'])
corr_volume = df['volume'].corr(df['price'])
avg_by_cut  = df.groupby('cut')['price'].mean().sort_values(ascending=False)
avg_by_color= df.groupby('color')['price'].mean().sort_values(ascending=False)

print(f"  Carat vs Price correlation  : {corr_carat:.3f}")
print(f"  Volume vs Price correlation : {corr_volume:.3f}")
print(f"  Most expensive cut          : {avg_by_cut.index[0]} (${avg_by_cut.iloc[0]:,.0f} avg)")
print(f"  Most expensive color        : {avg_by_color.index[0]} (${avg_by_color.iloc[0]:,.0f} avg)")
print(f"  Ideal cut avg price         : ${df[df['cut']=='Ideal']['price'].mean():,.0f}")
print(f"  Fair cut avg price          : ${df[df['cut']=='Fair']['price'].mean():,.0f}")

# ─────────────────────────────────────────
# 5. EDA DASHBOARD
# ─────────────────────────────────────────
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle('Diamond Price Analysis Dashboard', fontsize=20,
             fontweight='bold', color='#1A237E', y=1.01)

# Chart 1: Price Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['price'], bins=50, color='#5C6BC0', edgecolor='white', alpha=0.85)
ax1.axvline(df['price'].mean(), color='#E53935', linestyle='--',
            linewidth=2, label=f"Mean: ${df['price'].mean():,.0f}")
ax1.set_title('Price Distribution', fontweight='bold')
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Count')
ax1.legend(fontsize=9)

# Chart 2: Carat vs Price (scatter)
ax2 = fig.add_subplot(gs[0, 1])
sample = df.sample(2000, random_state=42)
scatter = ax2.scatter(sample['carat'], sample['price'],
                      c=sample['cut_score'], cmap='RdYlGn',
                      alpha=0.4, s=8)
ax2.set_title('Carat vs Price (colored by Cut)', fontweight='bold')
ax2.set_xlabel('Carat')
ax2.set_ylabel('Price ($)')
plt.colorbar(scatter, ax=ax2, label='Cut Quality →')

# Chart 3: Avg Price by Cut
ax3 = fig.add_subplot(gs[0, 2])
cut_order_list = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
avg_cut = df.groupby('cut')['price'].mean()[cut_order_list]
colors  = ['#EF5350','#FF7043','#FFA726','#66BB6A','#26C6DA']
bars = ax3.bar(avg_cut.index, avg_cut.values, color=colors, edgecolor='white')
ax3.set_title('Avg Price by Cut Quality', fontweight='bold')
ax3.set_xlabel('Cut')
ax3.set_ylabel('Avg Price ($)')
ax3.set_xticklabels(cut_order_list, rotation=15)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'${bar.get_height():,.0f}', ha='center', fontsize=8, fontweight='bold')

# Chart 4: Avg Price by Color
ax4 = fig.add_subplot(gs[1, 0])
color_order_list = ['D','E','F','G','H','I','J']
avg_color = df.groupby('color')['price'].mean()[color_order_list]
ax4.bar(avg_color.index, avg_color.values,
        color='#7E57C2', edgecolor='white', alpha=0.85)
ax4.set_title('Avg Price by Color (D=best, J=worst)', fontweight='bold')
ax4.set_xlabel('Color Grade')
ax4.set_ylabel('Avg Price ($)')

# Chart 5: Avg Price by Clarity
ax5 = fig.add_subplot(gs[1, 1])
clarity_list = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
avg_clarity = df.groupby('clarity')['price'].mean()[clarity_list]
ax5.bar(avg_clarity.index, avg_clarity.values,
        color='#26A69A', edgecolor='white', alpha=0.85)
ax5.set_title('Avg Price by Clarity', fontweight='bold')
ax5.set_xlabel('Clarity Grade')
ax5.set_ylabel('Avg Price ($)')
ax5.set_xticklabels(clarity_list, rotation=20)

# Chart 6: Correlation Heatmap
ax6 = fig.add_subplot(gs[1, 2])
num_cols = ['carat','depth','table','price','volume','cut_score','color_score','clarity_score']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax6, cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.5, annot_kws={'size': 7}, center=0)
ax6.set_title('Feature Correlation Heatmap', fontweight='bold')
ax6.tick_params(axis='x', rotation=30, labelsize=7)
ax6.tick_params(axis='y', rotation=0, labelsize=7)

plt.savefig('/mnt/user-data/outputs/diamond_eda_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("\n  EDA dashboard saved.")

# ─────────────────────────────────────────
# 6. MACHINE LEARNING — PRICE PREDICTION
# ─────────────────────────────────────────
print(f"\n[ STEP 5 ] MACHINE LEARNING — PRICE PREDICTION")

features = ['carat', 'depth', 'table', 'volume',
            'cut_score', 'color_score', 'clarity_score']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Model 1: Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_r2   = r2_score(y_test, lr_pred)
lr_mae  = mean_absolute_error(y_test, lr_pred)

# Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2   = r2_score(y_test, rf_pred)
rf_mae  = mean_absolute_error(y_test, rf_pred)

# Model 3: Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_r2   = r2_score(y_test, gb_pred)
gb_mae  = mean_absolute_error(y_test, gb_pred)

print(f"\n  Linear Regression  → R²: {lr_r2:.4f} | MAE: ${lr_mae:,.0f}")
print(f"  Random Forest      → R²: {rf_r2:.4f} | MAE: ${rf_mae:,.0f}")
print(f"  Gradient Boosting  → R²: {gb_r2:.4f} | MAE: ${gb_mae:,.0f}")

best_name = max([('Linear Regression', lr_r2),
                 ('Random Forest', rf_r2),
                 ('Gradient Boosting', gb_r2)], key=lambda x: x[1])
print(f"\n  Best Model: {best_name[0]} (R² = {best_name[1]:.4f})")

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(f"\n  Top 3 Price Predictors:")
for feat, imp in feat_imp.head(3).items():
    print(f"    {feat:15s}: {imp*100:.1f}%")

# ─────────────────────────────────────────
# 7. ML RESULTS DASHBOARD
# ─────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.patch.set_facecolor('#F8F9FA')
fig2.suptitle('Diamond Price — ML Results', fontsize=16,
              fontweight='bold', color='#1A237E')

# Chart 1: Actual vs Predicted (best model - RF)
sample_idx = np.random.choice(len(y_test), 500, replace=False)
axes[0].scatter(np.array(y_test)[sample_idx],
                rf_pred[sample_idx],
                alpha=0.3, color='#5C6BC0', s=10)
max_val = max(y_test.max(), rf_pred.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect fit')
axes[0].set_title(f'Actual vs Predicted\n(Random Forest, R²={rf_r2:.3f})', fontweight='bold')
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].legend()

# Chart 2: Model Comparison
models_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
r2_scores    = [lr_r2, rf_r2, gb_r2]
mae_scores   = [lr_mae, rf_mae, gb_mae]
bars = axes[1].bar(models_names, r2_scores,
                   color=['#42A5F5', '#66BB6A', '#FFA726'],
                   edgecolor='white', width=0.4)
axes[1].set_title('Model R² Comparison\n(higher = better)', fontweight='bold')
axes[1].set_ylabel('R² Score')
axes[1].set_ylim(0, 1.05)
for bar, score in zip(bars, r2_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', fontweight='bold', fontsize=10)

# Chart 3: Feature Importance
axes[2].barh(feat_imp.index[::-1], feat_imp.values[::-1],
             color='#7E57C2', edgecolor='white')
axes[2].set_title('Feature Importance\n(Random Forest)', fontweight='bold')
axes[2].set_xlabel('Importance Score')
for i, v in enumerate(feat_imp.values[::-1]):
    axes[2].text(v + 0.002, i, f'{v*100:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/diamond_ml_results.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')

print(f"\n[ STEP 6 ] ML dashboard saved.")

