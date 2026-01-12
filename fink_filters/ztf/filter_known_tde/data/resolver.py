import requests
import pandas as pd

APIURL = 'https://api.ztf.fink-portal.org'

data = {'name': [], 'ra': [], 'dec': []}

# Hammerstein TDEs
TDEs_hammerstein = pd.read_fwf('Table1_Hammerstein', skiprows=34, header=None)
tns_names = TDEs_hammerstein[0].to_list()

unknown = []
for name in tns_names:
    r = requests.post(
        '{}/api/v1/resolver'.format(APIURL),
        json={'resolver': 'simbad', 'name': name}
    )
    if r.json() == []:
        r = requests.post(
            '{}/api/v1/resolver'.format(APIURL),
            json={'resolver': 'tns', 'name': name}
        )

    if r.json() == []:
        unknown.append(name)
    else:
        data['name'].append(r.json()[0]['oname'])
        data['ra'].append(r.json()[0]['jradeg'])
        data['dec'].append(r.json()[0]['jdedeg'])
    # print(name, r.json())

# Gezari TDEs
pdf = pd.read_csv('TDElist_Gezari2021.tsv', sep='\t', header=None)
names = pdf[0].values

for name in names:
    if '/' in name:
        name = name.split('/')[0]
    r = requests.post(
        '{}/api/v1/resolver'.format(APIURL),
        json={'resolver': 'simbad', 'name': name}
    )
    if r.json() == []:
        r = requests.post(
            '{}/api/v1/resolver'.format(APIURL),
            json={'resolver': 'tns', 'name': name}
        )

    if r.json() == []:
        unknown.append(name)
    else:
        data['name'].append(r.json()[0]['oname'])
        data['ra'].append(r.json()[0]['jradeg'])
        data['dec'].append(r.json()[0]['jdedeg'])
    # print(name, r.json())
print('Not found: {}'.format(unknown))

pdf_final = pd.DataFrame.from_dict(data)
pdf_final = pdf_final.drop_duplicates()
print('Number of TDE found: {}/{}'.format(len(pdf_final), len(pdf_final) + len(unknown)))
pdf_final.to_parquet('tde.parquet')
