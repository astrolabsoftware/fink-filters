import pandas as pd

pdf1 = pd.read_csv("symbiotic_stars_catalog.txt")
pdf1["source"] = "symbiotic_stars"
pdf2 = pd.read_csv("cataclysmic_variables_catalog.txt")
pdf2["source"] = "cataclysmic_variables"

pdf = pd.concat((pdf1, pdf2))
print(pdf)
pdf.to_parquet('symbiotic_and_cataclysmic.parquet', index=None)
