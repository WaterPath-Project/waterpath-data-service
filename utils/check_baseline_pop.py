import pandas as pd

df = pd.read_csv('waterpath_data_service/data/kampala_dataset/default/human_emissions.csv')
print(f'Total baseline population: {df["population"].sum():.0f}')
print(f'Mean per area: {df["population"].mean():.0f}')
print(f'Number of areas: {len(df)}')
print(f'\nFirst 10 populations:')
print(df[['subarea', 'population']].head(10))
