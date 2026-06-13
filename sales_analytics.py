import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. GENERATE REALISTIC SALES DATASET
# ─────────────────────────────────────────
np.random.seed(42)
n = 1000

regions     = ['North', 'South', 'East', 'West']
categories  = ['Electronics', 'Clothing', 'Furniture', 'Grocery', 'Sports']
products    = {
    'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones'],
    'Clothing':    ['T-Shirt', 'Jeans', 'Jacket', 'Shoes'],
    'Furniture':   ['Chair', 'Table', 'Sofa', 'Shelf'],
    'Grocery':     ['Vegetables', 'Dairy', 'Snacks', 'Beverages'],
    'Sports':      ['Cricket Bat', 'Football', 'Yoga Mat', 'Dumbbells'],
}
base_prices = {
    'Laptop':50000,'Phone':25000,'Tablet':18000,'Headphones':3000,
    'T-Shirt':800,'Jeans':1500,'Jacket':3000,'Shoes':2500,
    'Chair':5000,'Table':8000,'Sofa':20000,'Shelf':4000,
    'Vegetables':200,'Dairy':150,'Snacks':100,'Beverages':80,
    'Cricket Bat':1200,'Football':800,'Yoga Mat':600,'Dumbbells':1500,
}

dates      = pd.date_range('2023-01-01', '2023-12-31', periods=n)
cat_list   = np.random.choice(categories, n)
prod_list  = [np.random.choice(products[c]) for c in cat_list]
price_list = [int(base_prices[p] * np.random.uniform(0.85, 1.15)) for p in prod_list]
qty_list   = np.random.randint(1, 6, n)

df = pd.DataFrame({
    'order_id':   range(1001, 1001 + n),
    'date':       dates,
    'region':     np.random.choice(regions, n),
    'category':   cat_list,
    'product':    prod_list,
    'quantity':   qty_list,
    'unit_price': price_list,
    'discount':   np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n),
    'customer_age': np.random.randint(18, 65, n),
})

# Inject missing values and duplicates for cleaning demo
df.loc[np.random.choice(df.index, 30), 'quantity']   = np.nan
df.loc[np.random.choice(df.index, 20), 'unit_price'] = np.nan
df.loc[np.random.choice(df.index, 15), 'region']     = np.nan
df = pd.concat([df, df.sample(10)], ignore_index=True)  # duplicate rows

print("=" * 55)
print("         SALES ANALYTICS PROJECT — NAVEEN")
print("=" * 55)

# ─────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────
print("\n[ STEP 1 ] RAW DATA OVERVIEW")
print(f"  Shape         : {df.shape}")
print(f"  Duplicates    : {df.duplicated().sum()}")
print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Region fillna
df['region'].fillna(df['region'].mode()[0], inplace=True)

# Fix dtypes
df['quantity']   = df['quantity'].fillna(df['quantity'].median()).astype(int)
df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median()).astype(int)
df['date']       = pd.to_datetime(df['date'])
df['month']      = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['month_order']= df['date'].dt.month

# Feature engineering
df['revenue']    = df['quantity'] * df['unit_price'] * (1 - df['discount'])
df['revenue']    = df['revenue'].round(2)

print(f"\n[ STEP 2 ] AFTER CLEANING")
print(f"  Shape         : {df.shape}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Total Revenue : ₹{df['revenue'].sum():,.0f}")

# ─────────────────────────────────────────
# 3. EDA — KEY INSIGHTS
# ─────────────────────────────────────────
print("\n[ STEP 3 ] KEY INSIGHTS")

top_cat   = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
top_reg   = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
monthly   = df.groupby(['month_order','month_name'])['revenue'].sum().reset_index().sort_values('month_order')
top_prod  = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)

print(f"\n  Top Category  : {top_cat.index[0]} (₹{top_cat.iloc[0]:,.0f})")
print(f"  Top Region    : {top_reg.index[0]} (₹{top_reg.iloc[0]:,.0f})")
print(f"  Best Month    : {monthly.loc[monthly['revenue'].idxmax(), 'month_name']} (₹{monthly['revenue'].max():,.0f})")
print(f"  Top Product   : {top_prod.index[0]} (₹{top_prod.iloc[0]:,.0f})")

# ─────────────────────────────────────────
# 4. DASHBOARD — 6 CHARTS
# ─────────────────────────────────────────
palette = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0']
sns.set_style("whitegrid")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

fig.suptitle('Sales Analytics Dashboard — 2023', fontsize=20, fontweight='bold',
             color='#1A237E', y=1.01)

# Chart 1: Monthly Revenue Trend
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(monthly['month_name'], monthly['revenue'] / 1e6,
         marker='o', color='#2196F3', linewidth=2.5, markersize=7)
ax1.fill_between(range(len(monthly)), monthly['revenue'] / 1e6,
                 alpha=0.15, color='#2196F3')
ax1.set_title('Monthly Revenue Trend', fontweight='bold', fontsize=13)
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (₹ Millions)')
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['month_name'], rotation=45)

# Chart 2: Revenue by Region (pie)
ax2 = fig.add_subplot(gs[0, 2])
ax2.pie(top_reg.values, labels=top_reg.index, autopct='%1.1f%%',
        colors=palette, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax2.set_title('Revenue by Region', fontweight='bold', fontsize=13)

# Chart 3: Revenue by Category (bar)
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(top_cat.index, top_cat.values / 1e6, color=palette, edgecolor='white')
ax3.set_title('Revenue by Category', fontweight='bold', fontsize=13)
ax3.set_xlabel('Category')
ax3.set_ylabel('Revenue (₹ Millions)')
ax3.set_xticklabels(top_cat.index, rotation=20, ha='right')
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f'₹{bar.get_height():.1f}M',
             ha='center', va='bottom', fontsize=8)

# Chart 4: Top 5 Products by Revenue
ax4 = fig.add_subplot(gs[1, 1])
ax4.barh(top_prod.index[::-1], top_prod.values[::-1] / 1e6,
         color='#4CAF50', edgecolor='white')
ax4.set_title('Top 5 Products by Revenue', fontweight='bold', fontsize=13)
ax4.set_xlabel('Revenue (₹ Millions)')
for i, v in enumerate(top_prod.values[::-1]):
    ax4.text(v / 1e6 + 0.02, i, f'₹{v/1e6:.1f}M', va='center', fontsize=8)

# Chart 5: Orders by Category (count)
ax5 = fig.add_subplot(gs[1, 2])
order_counts = df['category'].value_counts()
sns.barplot(x=order_counts.values, y=order_counts.index,
            palette=palette, ax=ax5)
ax5.set_title('Order Count by Category', fontweight='bold', fontsize=13)
ax5.set_xlabel('Number of Orders')
ax5.set_ylabel('')

plt.savefig('/mnt/user-data/outputs/sales_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print("\n[ STEP 4 ] Dashboard saved: sales_dashboard.png")

