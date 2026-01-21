"""
Temporary script to anonymize test data.
- Changes year from 2024 to 2022 (2 years backwards)
- Shifts all timestamps by exactly 6 days
- Anonymizes heights: rounds to nearest 10 while preserving distinct height count
"""

import pandas as pd
from datetime import timedelta
from collections import defaultdict

# Input and output file paths
input_file = r"H:\Repositories\local_repositories\TACT\tact\example\data\tact-test-data.csv"
output_file = r"H:\Repositories\local_repositories\TACT\tact\example\data\tact-test-data-anonymized.csv"

print(f"Reading data from: {input_file}")

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Loaded {len(df)} rows")
print(f"First few timestamps before anonymization:")
print(df['timestamp'].head(10).tolist())

# Parse timestamps - the format appears to be "M/D/YYYY H:MM"
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')

# Apply anonymization:
# 1. Subtract 2 years (730 days)
# 2. Add 6 days
# Net effect: subtract 2 years and add 6 days = subtract 724 days
df['timestamp'] = df['timestamp'] - timedelta(days=730) + timedelta(days=6)

# Convert back to the original format
df['timestamp'] = df['timestamp'].dt.strftime('%m/%d/%Y %H:%M')

print(f"\nFirst few timestamps after anonymization:")
print(df['timestamp'].head(10).tolist())

# Anonymize heights
print(f"\nAnonymizing heights...")
print(f"Original unique heights: {sorted(df['meas_height'].unique())}")
print(f"Number of distinct heights: {df['meas_height'].nunique()}")

# Get all unique heights sorted
unique_heights = sorted(df['meas_height'].unique())

# Group heights by what they would round to (nearest 10)
height_groups = defaultdict(list)
for height in unique_heights:
    rounded_base = round(height / 10) * 10
    height_groups[rounded_base].append(height)

# Create mapping: for each group, assign heights to different 10-interval values
height_mapping = {}
used_values = set()

for rounded_base in sorted(height_groups.keys()):
    heights_in_group = sorted(height_groups[rounded_base])
    
    for i, height in enumerate(heights_in_group):
        # Try to assign to rounded_base, rounded_base+5, rounded_base+10, etc.
        candidate = rounded_base + (i * 5)
        
        # If candidate is already used, keep incrementing by 5 until we find an unused value
        while candidate in used_values:
            candidate += 5
        
        height_mapping[height] = candidate
        used_values.add(candidate)

print(f"\nHeight mapping (original -> anonymized):")
for orig, new in sorted(height_mapping.items()):
    print(f"  {orig} -> {new}")

# Apply the mapping
df['meas_height'] = df['meas_height'].map(height_mapping)

print(f"\nAnonymized unique heights: {sorted(df['meas_height'].unique())}")
print(f"Number of distinct heights after anonymization: {df['meas_height'].nunique()}")

# Save to new file
print(f"\nSaving anonymized data to: {output_file}")
df.to_csv(output_file, index=False)

print(f"Anonymization complete! Output saved to: {output_file}")
print(f"Total rows processed: {len(df)}")
