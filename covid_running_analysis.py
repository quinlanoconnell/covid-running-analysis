import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================

print("Loading data...")
# Load running data
df_2019 = pd.read_csv('run_2019_m_sample.csv')
df_2020 = pd.read_csv('run_2020_m_sample.csv')

# Combine years
df_running = pd.concat([df_2019, df_2020], ignore_index=True)
print(f"✓ Combined running data: {len(df_running):,} rows")

# Load stringency index
df_stringency = pd.read_csv('covid-stringency-index.csv')
print(f"✓ Loaded stringency data: {len(df_stringency):,} rows")

# ============================================
# FIX: DIAGNOSE AND RESOLVE COUNTRY NAME MISMATCHES
# ============================================

print("\n" + "="*60)
print("DIAGNOSING COUNTRY NAME MISMATCHES")
print("="*60)

# Get unique countries from both datasets
running_countries = set(df_running['country'].unique())
stringency_countries = set(df_stringency['Entity'].unique())

print(f"\nCountries in running data: {len(running_countries)}")
print(f"Countries in stringency data: {len(stringency_countries)}")

# Find countries in running data NOT in stringency data
missing_in_stringency = running_countries - stringency_countries
print(f"\n🚨 Countries in running data but NOT in stringency data: {len(missing_in_stringency)}")
print("Top mismatches by athlete count:")

# Count athletes per missing country
missing_countries_count = df_running[df_running['country'].isin(missing_in_stringency)]['country'].value_counts()
print(missing_countries_count.head(15))

# Print exact names from stringency data that might be matches
print("\n--- Checking for similar names in stringency data ---")
for missing in list(missing_countries_count.head(10).index):
    print(f"\n'{missing}' not found. Possible matches in stringency data:")
    # Find similar names (containing part of the name)
    possible_matches = [s for s in stringency_countries if missing.lower() in s.lower() or s.lower() in missing.lower()]
    if possible_matches:
        for match in possible_matches[:3]:
            print(f"  - '{match}'")
    else:
        print("  - No obvious matches found")

# ============================================
# CREATE COUNTRY NAME MAPPING
# ============================================

print("\n" + "="*60)
print("CREATING COUNTRY NAME MAPPING")
print("="*60)

# Create a mapping dictionary for mismatched names
# You'll need to customize this based on the diagnostic output above
country_mapping = {
    'United States': 'United States',  # Check if it's in stringency as this or 'USA'
    'Netherlands': 'Netherlands',
    'Sweden': 'Sweden',
    'Thailand': 'Thailand',
    'Mexico': 'Mexico',
    'Switzerland': 'Switzerland',
    'Denmark': 'Denmark',
    'Chile': 'Chile',
    'Ireland': 'Ireland',
    'Ukraine': 'Ukraine'
}

# Let's check what these countries are called in stringency data
print("\nChecking stringency data for these countries...")
for country in ['United States', 'Netherlands', 'Sweden', 'Thailand', 'Mexico']:
    matches = df_stringency[df_stringency['Entity'].str.contains(country, case=False, na=False)]['Entity'].unique()
    if len(matches) > 0:
        print(f"'{country}' → Found as: {matches}")
    else:
        print(f"'{country}' → NOT FOUND")

# ============================================
# APPLY MAPPING AND RE-MERGE
# ============================================

print("\n" + "="*60)
print("RE-MERGING WITH CORRECTED COUNTRY NAMES")
print("="*60)

# First, let's see what the actual country names are in stringency
print("\nAll country names in stringency data (first 30):")
print(sorted(stringency_countries)[:30])

# ============================================
# STEP 2: PREPARE STRINGENCY DATA (DAILY → MONTHLY)
# ============================================

print("\nPreparing stringency data...")
# Convert date to datetime
df_stringency['Date'] = pd.to_datetime(df_stringency['Date'])

# Extract year-month
df_stringency['year_month'] = df_stringency['Date'].dt.to_period('M')

# Calculate monthly average stringency by country
df_stringency_monthly = df_stringency.groupby(['Entity', 'year_month']).agg({
    'stringency_index': 'mean'
}).reset_index()

# Convert period back to string for merging
df_stringency_monthly['year_month'] = df_stringency_monthly['year_month'].astype(str)
print(f"✓ Aggregated to monthly: {len(df_stringency_monthly):,} country-months")

# ============================================
# STEP 3: PREPARE RUNNING DATA FOR MERGE
# ============================================

print("\nPreparing running data...")
# Convert datetime to datetime object
df_running['datetime'] = pd.to_datetime(df_running['datetime'])

# Extract year-month in same format
df_running['year_month'] = df_running['datetime'].dt.to_period('M').astype(str)

# Create COVID period dummy (March 2020 onwards)
df_running['covid_period'] = (df_running['datetime'] >= '2020-03-01').astype(int)

# Create time trend (months since start)
start_date = df_running['datetime'].min()
df_running['time_trend'] = ((df_running['datetime'].dt.year - start_date.year) * 12 +
                              (df_running['datetime'].dt.month - start_date.month))

# Extract year and month
df_running['year'] = df_running['datetime'].dt.year
df_running['month'] = df_running['datetime'].dt.month

print(f"✓ Created time variables")

# ============================================
# STEP 4: MERGE RUNNING DATA WITH STRINGENCY INDEX
# ============================================

print("\nMerging datasets...")
# Merge on country and month
df_merged = df_running.merge(
    df_stringency_monthly,
    left_on=['country', 'year_month'],
    right_on=['Entity', 'year_month'],
    how='left'
)

print(f"✓ Merged data: {len(df_merged):,} rows")

# Check merge success
merge_rate = (df_merged['stringency_index'].notna().sum() / len(df_merged) * 100)
print(f"  - Stringency index matched: {merge_rate:.1f}%")

# For 2019, fill stringency with 0 (no COVID restrictions)
df_merged.loc[df_merged['year'] == 2019, 'stringency_index'] = 0
df_merged['stringency_index'] = df_merged['stringency_index'].fillna(0)

# ============================================
# STEP 5: CREATE ANALYSIS VARIABLES
# ============================================

print("\nCreating analysis variables...")

# Create binary active indicator
df_merged['active'] = (df_merged['distance'] > 0).astype(int)

# Create dummy variables for age groups and gender
df_merged['age_18_34'] = (df_merged['age_group'] == '18 - 34').astype(int)
df_merged['age_35_54'] = (df_merged['age_group'] == '35 - 54').astype(int)
df_merged['age_55_plus'] = (df_merged['age_group'] == '55 +').astype(int)
df_merged['female'] = (df_merged['gender'] == 'F').astype(int)

# Create interaction terms
df_merged['covid_x_age35_54'] = df_merged['covid_period'] * df_merged['age_35_54']
df_merged['covid_x_age55plus'] = df_merged['covid_period'] * df_merged['age_55_plus']
df_merged['covid_x_female'] = df_merged['covid_period'] * df_merged['female']

# Create stringency interactions
df_merged['stringency_x_age35_54'] = df_merged['stringency_index'] * df_merged['age_35_54']
df_merged['stringency_x_age55plus'] = df_merged['stringency_index'] * df_merged['age_55_plus']
df_merged['stringency_x_female'] = df_merged['stringency_index'] * df_merged['female']

print(f"✓ Analysis dataset ready: {len(df_merged):,} observations")

# ============================================
# STEP 6: DESCRIPTIVE STATISTICS
# ============================================

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

# Overall summary
print("\n--- Overall Running Statistics ---")
print(f"Total observations: {len(df_merged):,}")
print(f"Unique athletes: {df_merged['athlete'].nunique():,}")
print(f"Date range: {df_merged['datetime'].min().date()} to {df_merged['datetime'].max().date()}")

print("\n--- Activity Rates ---")
activity_by_year = df_merged.groupby('year')['active'].agg(['mean', 'count'])
activity_by_year['mean'] = activity_by_year['mean'] * 100
print(activity_by_year)

print("\n--- Average Distance (Active Months Only) ---")
active_df = df_merged[df_merged['active'] == 1]
dist_by_year = active_df.groupby('year')['distance'].agg(['mean', 'std', 'count'])
print(dist_by_year)

print("\n--- By Age Group (2019 vs 2020) ---")
age_comparison = df_merged.groupby(['year', 'age_group'])['distance'].mean().reset_index()
age_pivot = age_comparison.pivot(index='age_group', columns='year', values='distance')
age_pivot['change_%'] = ((age_pivot[2020] - age_pivot[2019]) / age_pivot[2019] * 100)
print(age_pivot)

print("\n--- By Gender (2019 vs 2020) ---")
gender_comparison = df_merged.groupby(['year', 'gender'])['distance'].mean().reset_index()
gender_pivot = gender_comparison.pivot(index='gender', columns='year', values='distance')
gender_pivot['change_%'] = ((gender_pivot[2020] - gender_pivot[2019]) / gender_pivot[2019] * 100)
print(gender_pivot)

# ============================================
# STEP 7: REGRESSION MODELS
# ============================================

print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# MODEL 1: Basic Time Trend + COVID Period
print("\n--- MODEL 1: Basic COVID Effect ---")
model1 = smf.ols(
    'distance ~ time_trend + covid_period',
    data=df_merged
).fit()
print(model1.summary().tables[1])

# MODEL 2: Add Stringency Index
print("\n--- MODEL 2: Adding Stringency Index ---")
model2 = smf.ols(
    'distance ~ time_trend + covid_period + stringency_index',
    data=df_merged
).fit()
print(model2.summary().tables[1])

# MODEL 3: Add Demographics
print("\n--- MODEL 3: Adding Demographics ---")
model3 = smf.ols(
    'distance ~ time_trend + covid_period + stringency_index + age_35_54 + age_55_plus + female',
    data=df_merged
).fit()
print(model3.summary().tables[1])

# MODEL 4: Full Model with Interactions (MAIN MODEL)
print("\n--- MODEL 4: FULL MODEL WITH INTERACTIONS ---")
model4 = smf.ols(
    '''distance ~ time_trend + covid_period + stringency_index + 
       age_35_54 + age_55_plus + female +
       covid_x_age35_54 + covid_x_age55plus + covid_x_female +
       stringency_x_age35_54 + stringency_x_age55plus + stringency_x_female''',
    data=df_merged
).fit()
print(model4.summary().tables[1])

# Print model comparison
print("\n--- MODEL COMPARISON ---")
print(f"{'Model':<30} {'R-squared':<12} {'Adj. R-squared':<15} {'N':<10}")
print("-" * 70)
print(f"{'1. Time + COVID':<30} {model1.rsquared:.4f}       {model1.rsquared_adj:.4f}          {int(model1.nobs)}")
print(f"{'2. + Stringency':<30} {model2.rsquared:.4f}       {model2.rsquared_adj:.4f}          {int(model2.nobs)}")
print(f"{'3. + Demographics':<30} {model3.rsquared:.4f}       {model3.rsquared_adj:.4f}          {int(model3.nobs)}")
print(f"{'4. + Interactions (FULL)':<30} {model4.rsquared:.4f}       {model4.rsquared_adj:.4f}          {int(model4.nobs)}")

# ============================================
# STEP 8: CALCULATE KEY EFFECTS
# ============================================

print("\n" + "="*60)
print("KEY FINDINGS FROM MAIN MODEL")
print("="*60)

# Extract coefficients
coefs = model4.params
pvals = model4.pvalues

print("\n--- Direct COVID Effects (vs. baseline 18-34 males) ---")
print(f"COVID period effect: {coefs['covid_period']:.2f} km (p={pvals['covid_period']:.4f})")
print(f"Stringency effect (per 10-point increase): {coefs['stringency_index']*10:.2f} km (p={pvals['stringency_index']:.4f})")

print("\n--- Demographic Main Effects ---")
print(f"Age 35-54: {coefs['age_35_54']:.2f} km (p={pvals['age_35_54']:.4f})")
print(f"Age 55+: {coefs['age_55_plus']:.2f} km (p={pvals['age_55_plus']:.4f})")
print(f"Female: {coefs['female']:.2f} km (p={pvals['female']:.4f})")

print("\n--- COVID × Demographics Interactions ---")
print(f"COVID × Age 35-54: {coefs['covid_x_age35_54']:.2f} km (p={pvals['covid_x_age35_54']:.4f})")
print(f"COVID × Age 55+: {coefs['covid_x_age55plus']:.2f} km (p={pvals['covid_x_age55plus']:.4f})")
print(f"COVID × Female: {coefs['covid_x_female']:.2f} km (p={pvals['covid_x_female']:.4f})")

print("\n--- Stringency × Demographics Interactions ---")
print(f"Stringency × Age 35-54: {coefs['stringency_x_age35_54']:.3f} km (p={pvals['stringency_x_age35_54']:.4f})")
print(f"Stringency × Age 55+: {coefs['stringency_x_age55plus']:.3f} km (p={pvals['stringency_x_age55plus']:.4f})")
print(f"Stringency × Female: {coefs['stringency_x_female']:.3f} km (p={pvals['stringency_x_female']:.4f})")

# ============================================
# SAVE RESULTS
# ============================================

# Save the merged dataset
df_merged.to_csv('analysis_dataset.csv', index=False)
print("\n✓ Saved analysis dataset to 'analysis_dataset.csv'")

# Save model results
with open('regression_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FULL REGRESSION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(model4.summary().as_text())

print("✓ Saved regression results to 'regression_results.txt'")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

# Add this diagnostic code:
print("\n--- Stringency Match Check ---")
missing_stringency = df_merged[
    (df_merged['year'] == 2020) & (df_merged['stringency_index'] == 0)
]['country'].value_counts()
print(f"\n2020 countries with missing stringency (top 10):")
print(missing_stringency.head(10))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# ============================================
# LOAD YOUR ANALYSIS DATASET
# ============================================

df_merged = pd.read_csv('analysis_dataset.csv')
df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])

print("="*60)
print("MODEL 3: DEMOGRAPHICS + STRINGENCY (MAIN MODEL)")
print("="*60)

# ============================================
# RUN MODEL 3 (YOUR MAIN MODEL)
# ============================================

model3 = smf.ols(
    'distance ~ time_trend + covid_period + stringency_index + age_35_54 + age_55_plus + female',
    data=df_merged
).fit()

print("\n--- REGRESSION RESULTS ---")
print(model3.summary())

# Extract key coefficients for interpretation
coefs = model3.params
pvals = model3.pvalues
stderr = model3.bse

print("\n" + "="*60)
print("KEY FINDINGS (Model 3)")
print("="*60)

print("\n📊 MAIN EFFECTS:")
print(f"  Time trend: {coefs['time_trend']:.2f} km/month (p<0.001)")
print(f"  COVID period: {coefs['covid_period']:.2f} km (p<0.001)")
print(f"  Stringency (per 10 points): {coefs['stringency_index']*10:.2f} km (p<0.001)")

print("\n👥 DEMOGRAPHIC EFFECTS:")
print(f"  Age 35-54 vs 18-34: +{coefs['age_35_54']:.2f} km (p<0.001)")
print(f"  Age 55+ vs 18-34: +{coefs['age_55_plus']:.2f} km (p<0.001)")
print(f"  Female vs Male: {coefs['female']:.2f} km (p<0.001)")

print(f"\n📈 Model R-squared: {model3.rsquared:.4f}")
print(f"📊 Observations: {int(model3.nobs):,}")

# ============================================
# CALCULATE PREDICTED VALUES FOR VISUALIZATION
# ============================================

print("\n" + "="*60)
print("CREATING VISUALIZATION DATA")
print("="*60)

# Create monthly aggregates for visualization
monthly_actual = df_merged.groupby(['datetime', 'age_group']).agg({
    'distance': 'mean',
    'active': 'mean',
    'stringency_index': 'mean',
    'athlete': 'count'
}).reset_index()

monthly_actual['activity_pct'] = monthly_actual['active'] * 100

print(f"✓ Created monthly aggregates: {len(monthly_actual)} month-group combinations")

# ============================================
# VISUALIZATION 1: MAIN STORY PLOT
# Time series showing running distance + stringency overlay
# ============================================

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot average distance by age group
for age in ['18 - 34', '35 - 54', '55 +']:
    data_age = monthly_actual[monthly_actual['age_group'] == age]
    ax1.plot(data_age['datetime'], data_age['distance'],
             marker='o', linewidth=2.5, markersize=6,
             label=f'Age {age}', alpha=0.8)

ax1.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--',
            linewidth=2, alpha=0.7, label='COVID Start')

ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
ax1.set_ylabel('Average Monthly Distance (km)', fontsize=13, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Running Distance During COVID-19 by Age Group',
              fontsize=16, fontweight='bold', pad=20)

# Add stringency index on secondary axis
ax2 = ax1.twinx()
data_2020 = monthly_actual[monthly_actual['datetime'] >= '2020-01-01'].groupby('datetime')['stringency_index'].mean()
ax2.fill_between(data_2020.index, 0, data_2020.values,
                 color='gray', alpha=0.2, label='Lockdown Stringency')
ax2.set_ylabel('Stringency Index (0-100)', fontsize=13, fontweight='bold', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('slide1_main_story.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide1_main_story.png")
plt.close()

# ============================================
# VISUALIZATION 2: REGRESSION COEFFICIENTS PLOT
# Clean visualization of your key findings
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT PANEL: COVID & Stringency Effects
covid_effects = pd.DataFrame({
    'Variable': ['Time Trend\n(per month)', 'COVID Period\n(vs 2019)', 'Stringency\n(per 10 points)'],
    'Coefficient': [coefs['time_trend'], coefs['covid_period'], coefs['stringency_index']*10],
    'SE': [stderr['time_trend'], stderr['covid_period'], stderr['stringency_index']*10]
})

colors = ['#d62728' if x < 0 else '#2ca02c' for x in covid_effects['Coefficient']]
bars1 = ax1.barh(covid_effects['Variable'], covid_effects['Coefficient'],
                 color=colors, alpha=0.7, height=0.6)

# Add error bars
ax1.errorbar(covid_effects['Coefficient'], range(len(covid_effects)),
             xerr=covid_effects['SE']*1.96, fmt='none', color='black',
             capsize=5, linewidth=2)

ax1.axvline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Effect on Monthly Distance (km)', fontsize=13, fontweight='bold')
ax1.set_title('COVID & Time Effects', fontsize=14, fontweight='bold')
ax1.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (coef, var) in enumerate(zip(covid_effects['Coefficient'], covid_effects['Variable'])):
    ax1.text(coef, i, f'  {coef:.2f}',
             va='center', ha='left' if coef > 0 else 'right',
             fontweight='bold', fontsize=11)

# RIGHT PANEL: Demographic Effects
demo_effects = pd.DataFrame({
    'Variable': ['Age 35-54\n(vs 18-34)', 'Age 55+\n(vs 18-34)', 'Female\n(vs Male)'],
    'Coefficient': [coefs['age_35_54'], coefs['age_55_plus'], coefs['female']],
    'SE': [stderr['age_35_54'], stderr['age_55_plus'], stderr['female']]
})

colors2 = ['#d62728' if x < 0 else '#2ca02c' for x in demo_effects['Coefficient']]
bars2 = ax2.barh(demo_effects['Variable'], demo_effects['Coefficient'],
                 color=colors2, alpha=0.7, height=0.6)

# Add error bars
ax2.errorbar(demo_effects['Coefficient'], range(len(demo_effects)),
             xerr=demo_effects['SE']*1.96, fmt='none', color='black',
             capsize=5, linewidth=2)

ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Effect on Monthly Distance (km)', fontsize=13, fontweight='bold')
ax2.set_title('Demographic Effects', fontsize=14, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (coef, var) in enumerate(zip(demo_effects['Coefficient'], demo_effects['Variable'])):
    ax2.text(coef, i, f'  {coef:.2f}',
             va='center', ha='left' if coef > 0 else 'right',
             fontweight='bold', fontsize=11)

plt.suptitle('Regression Results: How COVID Changed Running Behavior',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('slide2_regression_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide2_regression_results.png")
plt.close()

# ============================================
# VISUALIZATION 3: ACTIVITY RATE (BONUS)
# Percentage of athletes running each month
# ============================================

fig, ax = plt.subplots(figsize=(14, 7))

for age in ['18 - 34', '35 - 54', '55 +']:
    data_age = monthly_actual[monthly_actual['age_group'] == age]
    ax.plot(data_age['datetime'], data_age['activity_pct'],
            marker='o', linewidth=2.5, markersize=6,
            label=f'Age {age}', alpha=0.8)

ax.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--',
           linewidth=2, alpha=0.7, label='COVID Start')

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Athletes Running (%)', fontsize=13, fontweight='bold')
ax.set_ylim(70, 95)
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title('Running Participation Rate During COVID-19',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('bonus_activity_rate.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bonus_activity_rate.png")
plt.close()

# ============================================
# CREATE SUMMARY TABLE FOR SLIDE 2
# ============================================

summary_table = pd.DataFrame({
    'Variable': [
        'Time Trend (per month)',
        'COVID Period',
        'Stringency (per 10-point increase)',
        'Age 35-54',
        'Age 55+',
        'Female'
    ],
    'Coefficient': [
        coefs['time_trend'],
        coefs['covid_period'],
        coefs['stringency_index'] * 10,
        coefs['age_35_54'],
        coefs['age_55_plus'],
        coefs['female']
    ],
    'Std Error': [
        stderr['time_trend'],
        stderr['covid_period'],
        stderr['stringency_index'] * 10,
        stderr['age_35_54'],
        stderr['age_55_plus'],
        stderr['female']
    ],
    'p-value': [
        pvals['time_trend'],
        pvals['covid_period'],
        pvals['stringency_index'],
        pvals['age_35_54'],
        pvals['age_55_plus'],
        pvals['female']
    ]
})

summary_table['Significance'] = summary_table['p-value'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

print("\n" + "="*60)
print("REGRESSION TABLE FOR YOUR SLIDE")
print("="*60)
print(summary_table.to_string(index=False))

summary_table.to_csv('regression_table_for_slide.csv', index=False)
print("\n✓ Saved: regression_table_for_slide.csv")

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print("\nCreated files:")
print("  1. slide1_main_story.png - Time series showing running trends")
print("  2. slide2_regression_results.png - Coefficient plot")
print("  3. bonus_activity_rate.png - Alternative view (participation %)")
print("  4. regression_table_for_slide.csv - Clean table for your slide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df_merged = pd.read_csv('analysis_dataset.csv')
df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("Creating exploratory visualizations...\n")

# ============================================
# 1. PIE CHART: Athletes by Country (Top 10)
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Get unique athletes per country
athletes_per_country = df_merged.groupby('country')['athlete'].nunique().sort_values(ascending=False)

# Top 10 countries + "Other"
top10_countries = athletes_per_country.head(10)
other_count = athletes_per_country[10:].sum()

# Combine for pie chart
pie_data = pd.concat([top10_countries, pd.Series({'Other': other_count})])

colors = plt.cm.Set3(range(len(pie_data)))
wedges, texts, autotexts = ax1.pie(pie_data,
                                   labels=pie_data.index,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90,
                                   textprops={'fontsize': 10})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax1.set_title('Athletes by Country\n(Top 10 + Other)',
              fontsize=14, fontweight='bold', pad=20)

# ============================================
# 2. PIE CHART: Athletes by Age Group
# ============================================

athletes_per_age = df_merged.groupby('age_group')['athlete'].nunique()

colors_age = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = ax2.pie(athletes_per_age,
                                   labels=athletes_per_age.index,
                                   autopct='%1.1f%%',
                                   colors=colors_age,
                                   startangle=90,
                                   textprops={'fontsize': 11})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax2.set_title('Athletes by Age Group',
              fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('exploratory_demographics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: exploratory_demographics.png")
plt.close()

# ============================================
# 3. BAR CHART: Top 15 Countries
# ============================================

fig, ax = plt.subplots(figsize=(12, 8))

top15 = athletes_per_country.head(15)
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top15)))

bars = ax.barh(range(len(top15)), top15.values, color=colors_bar, alpha=0.8)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index)
ax.invert_yaxis()
ax.set_xlabel('Number of Athletes', fontsize=13, fontweight='bold')
ax.set_title('Top 15 Countries by Number of Athletes',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (country, count) in enumerate(zip(top15.index, top15.values)):
    ax.text(count, i, f'  {count:,}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('exploratory_countries_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: exploratory_countries_bar.png")
plt.close()

# ============================================
# 4. STACKED BAR: Gender Distribution by Age
# ============================================

fig, ax = plt.subplots(figsize=(10, 7))

# Get counts by age and gender
gender_age = df_merged.groupby(['age_group', 'gender'])['athlete'].nunique().unstack()

# Calculate percentages
gender_age_pct = gender_age.div(gender_age.sum(axis=1), axis=0) * 100

# Create stacked bar
gender_age_pct.plot(kind='bar', stacked=True, ax=ax,
                    color=['#ff9999', '#6699ff'],
                    width=0.6, alpha=0.8)

ax.set_ylabel('Percentage', fontsize=13, fontweight='bold')
ax.set_xlabel('Age Group', fontsize=13, fontweight='bold')
ax.set_title('Gender Distribution by Age Group',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(title='Gender', labels=['Female', 'Male'],
          fontsize=11, title_fontsize=12)
ax.set_xticklabels(gender_age_pct.index, rotation=0)
ax.grid(axis='y', alpha=0.3)

# Add count labels
for i, age in enumerate(gender_age.index):
    female_pct = gender_age_pct.loc[age, 'F']
    male_pct = gender_age_pct.loc[age, 'M']

    female_count = gender_age.loc[age, 'F']
    male_count = gender_age.loc[age, 'M']

    ax.text(i, female_pct / 2, f'{female_count:,}\n({female_pct:.1f}%)',
            ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(i, female_pct + male_pct / 2, f'{male_count:,}\n({male_pct:.1f}%)',
            ha='center', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('exploratory_gender_age.png', dpi=300, bbox_inches='tight')
print("✓ Saved: exploratory_gender_age.png")
plt.close()

# ============================================
# 5. HISTOGRAM: Distribution of Running Distance
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: All data including zeros
axes[0].hist(df_merged['distance'], bins=50, color='steelblue',
             alpha=0.7, edgecolor='black')
axes[0].axvline(df_merged['distance'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Mean: {df_merged["distance"].mean():.1f} km')
axes[0].axvline(df_merged['distance'].median(), color='orange',
                linestyle='--', linewidth=2, label=f'Median: {df_merged["distance"].median():.1f} km')
axes[0].set_xlabel('Monthly Distance (km)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Frequency (athlete-months)', fontsize=13, fontweight='bold')
axes[0].set_title('Distribution of Running Distance\n(All Data)',
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Right panel: Active months only (distance > 0)
active_data = df_merged[df_merged['distance'] > 0]['distance']
axes[1].hist(active_data, bins=50, color='forestgreen',
             alpha=0.7, edgecolor='black')
axes[1].axvline(active_data.mean(), color='red',
                linestyle='--', linewidth=2, label=f'Mean: {active_data.mean():.1f} km')
axes[1].axvline(active_data.median(), color='orange',
                linestyle='--', linewidth=2, label=f'Median: {active_data.median():.1f} km')
axes[1].set_xlabel('Monthly Distance (km)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Frequency (athlete-months)', fontsize=13, fontweight='bold')
axes[1].set_title('Distribution of Running Distance\n(Active Months Only)',
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('exploratory_distance_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: exploratory_distance_distribution.png")
plt.close()

# ============================================
# 6. SUMMARY STATISTICS TABLE (Visual)
# ============================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

# Create summary statistics
summary_stats = []

# Overall stats
summary_stats.append(['DATASET OVERVIEW', '', ''])
summary_stats.append(['Total observations', f"{len(df_merged):,}", ''])
summary_stats.append(['Unique athletes', f"{df_merged['athlete'].nunique():,}", ''])
summary_stats.append(['Countries represented', f"{df_merged['country'].nunique()}", ''])
summary_stats.append(['Time period', '2019-01 to 2020-12', '24 months'])
summary_stats.append(['', '', ''])

# Demographics
summary_stats.append(['DEMOGRAPHICS', '', ''])
for age in ['18 - 34', '35 - 54', '55 +']:
    count = df_merged[df_merged['age_group'] == age]['athlete'].nunique()
    pct = count / df_merged['athlete'].nunique() * 100
    summary_stats.append([f'  {age}', f'{count:,}', f'{pct:.1f}%'])

female_count = df_merged[df_merged['gender'] == 'F']['athlete'].nunique()
male_count = df_merged[df_merged['gender'] == 'M']['athlete'].nunique()
summary_stats.append(['  Female', f'{female_count:,}', f'{female_count / (female_count + male_count) * 100:.1f}%'])
summary_stats.append(['  Male', f'{male_count:,}', f'{male_count / (female_count + male_count) * 100:.1f}%'])
summary_stats.append(['', '', ''])

# Running statistics
summary_stats.append(['RUNNING STATISTICS', '', ''])
summary_stats.append(['Average distance (all)', f"{df_merged['distance'].mean():.1f} km/month", ''])
summary_stats.append(['Average distance (active)',
                      f"{df_merged[df_merged['active'] == 1]['distance'].mean():.1f} km/month", ''])
summary_stats.append(['Activity rate 2019', f"{df_merged[df_merged['year'] == 2019]['active'].mean() * 100:.1f}%", ''])
summary_stats.append(['Activity rate 2020', f"{df_merged[df_merged['year'] == 2020]['active'].mean() * 100:.1f}%", ''])
summary_stats.append(['', '', ''])

# COVID stats
summary_stats.append(['COVID STRINGENCY (2020)', '', ''])
df_2020 = df_merged[df_merged['year'] == 2020]
summary_stats.append(['Average stringency',
                      f"{df_2020[df_2020['stringency_index'] > 0]['stringency_index'].mean():.1f}",
                      '(0-100 scale)'])
summary_stats.append(['Max stringency observed',
                      f"{df_2020['stringency_index'].max():.1f}", ''])
summary_stats.append(['Min stringency (2020)',
                      f"{df_2020[df_2020['stringency_index'] > 0]['stringency_index'].min():.1f}", ''])

# Create table
table = ax.table(cellText=summary_stats,
                 colWidths=[0.5, 0.3, 0.2],
                 cellLoc='left',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Style the table
for i, row in enumerate(summary_stats):
    if row[0] in ['DATASET OVERVIEW', 'DEMOGRAPHICS', 'RUNNING STATISTICS', 'COVID STRINGENCY (2020)']:
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white', fontsize=12)

plt.title('Dataset Summary Statistics',
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('exploratory_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: exploratory_summary_table.png")
plt.close()

# ============================================
# PRINT TEXT SUMMARY
# ============================================

print("\n" + "=" * 60)
print("STRINGENCY INDEX IN YOUR DATA")
print("=" * 60)

df_2020_stringency = df_merged[(df_merged['year'] == 2020) &
                               (df_merged['stringency_index'] > 0)]

print(f"\n2020 Stringency Statistics:")
print(f"  Mean: {df_2020_stringency['stringency_index'].mean():.1f}")
print(f"  Median: {df_2020_stringency['stringency_index'].median():.1f}")
print(f"  Min: {df_2020_stringency['stringency_index'].min():.1f}")
print(f"  Max: {df_2020_stringency['stringency_index'].max():.1f}")
print(f"  Std Dev: {df_2020_stringency['stringency_index'].std():.1f}")

print("\n" + "=" * 60)
print("EXPLORATORY VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nCreated files:")
print("  1. exploratory_demographics.png - Pie charts of country/age distribution")
print("  2. exploratory_countries_bar.png - Top 15 countries bar chart")
print("  3. exploratory_gender_age.png - Gender distribution by age")
print("  4. exploratory_distance_distribution.png - Histogram of running distances")
print("  5. exploratory_summary_table.png - Complete dataset summary")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Load data and run model
df_merged = pd.read_csv('analysis_dataset.csv')

model3 = smf.ols(
    'distance ~ time_trend + covid_period + stringency_index + age_35_54 + age_55_plus + female',
    data=df_merged
).fit()

coefs = model3.params
stderr = model3.bse

# ============================================
# CLEAN VISUALIZATION: REGRESSION COEFFICIENTS
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT PANEL: COVID & Stringency Effects
covid_effects = pd.DataFrame({
    'Variable': ['Time Trend\n(per month)', 'COVID Period\n(vs 2019)', 'Stringency\n(per 10 points)'],
    'Coefficient': [coefs['time_trend'], coefs['covid_period'], coefs['stringency_index']*10],
    'SE': [stderr['time_trend'], stderr['covid_period'], stderr['stringency_index']*10]
})

colors = ['#d62728' if x < 0 else '#2ca02c' for x in covid_effects['Coefficient']]
bars1 = ax1.barh(covid_effects['Variable'], covid_effects['Coefficient'],
                 color=colors, alpha=0.7, height=0.6)

# Add error bars
ax1.errorbar(covid_effects['Coefficient'], range(len(covid_effects)),
             xerr=covid_effects['SE']*1.96, fmt='none', color='black',
             capsize=5, linewidth=2)

ax1.axvline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Effect on Monthly Distance (km)', fontsize=13, fontweight='bold')
ax1.set_title('COVID & Time Effects', fontsize=14, fontweight='bold')
ax1.grid(True, axis='x', alpha=0.3)

# Simple labels INSIDE bars
for i, coef in enumerate(covid_effects['Coefficient']):
    # Position in middle of bar
    x_pos = coef / 2
    ax1.text(x_pos, i, f'{coef:.2f}',
            va='center', ha='center',
            fontweight='bold', fontsize=13,
            color='white')

# RIGHT PANEL: Demographic Effects
demo_effects = pd.DataFrame({
    'Variable': ['Age 35-54\n(vs 18-34)', 'Age 55+\n(vs 18-34)', 'Female\n(vs Male)'],
    'Coefficient': [coefs['age_35_54'], coefs['age_55_plus'], coefs['female']],
    'SE': [stderr['age_35_54'], stderr['age_55_plus'], stderr['female']]
})

colors2 = ['#d62728' if x < 0 else '#2ca02c' for x in demo_effects['Coefficient']]
bars2 = ax2.barh(demo_effects['Variable'], demo_effects['Coefficient'],
                 color=colors2, alpha=0.7, height=0.6)

# Add error bars
ax2.errorbar(demo_effects['Coefficient'], range(len(demo_effects)),
             xerr=demo_effects['SE']*1.96, fmt='none', color='black',
             capsize=5, linewidth=2)

ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Effect on Monthly Distance (km)', fontsize=13, fontweight='bold')
ax2.set_title('Demographic Effects', fontsize=14, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# Simple labels INSIDE bars
for i, coef in enumerate(demo_effects['Coefficient']):
    # Position in middle of bar
    x_pos = coef / 2
    ax2.text(x_pos, i, f'{coef:.2f}',
            va='center', ha='center',
            fontweight='bold', fontsize=13,
            color='white')

plt.suptitle('Regression Results: How COVID Changed Running Behavior',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('slide2_regression_results_clean.png', dpi=300, bbox_inches='tight')
print("✓ Saved: slide2_regression_results_clean.png")
plt.close()

print("\n✓ Clean version with simple inside labels created!")