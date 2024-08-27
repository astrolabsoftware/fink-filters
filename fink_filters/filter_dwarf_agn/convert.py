import pandas as pd

pdf = pd.read_csv('list_dwarfs_AGN_RADEC.txt', sep=r'\s+', header=0)
print(pdf)
print(pdf.info())
pdf.to_parquet('list_dwarfs_AGN_RADEC.parquet', index=None)
