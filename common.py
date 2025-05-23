DATA_EMB_DIC1 = {
    'bonanza': (7919,1973),
    'house1to10': (515, 1281),
    'senate1to10': (145, 1056),
    'review': (182, 304),
    'ml-1m': (6040, 3952),
    'ml-10m': (69878, 10677),
    'ml-32m': (200948, 292757),
    'amazon-book': (35736, 38121),
    'amazon-books-2018': (2930451, 15362619),
    'amazon-automotive-2018': (925387, 3873247),
    'amazon-arts-2018': (302809, 1579230),
    'amazon-cds-2018': (434060, 1944316)
}

DATA_EMB_DIC = {**DATA_EMB_DIC1}
for k in DATA_EMB_DIC1:
    for i in range(1, 6):
        DATA_EMB_DIC.update({
            f'{k}-{i}': DATA_EMB_DIC1[k]
        })
