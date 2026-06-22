import openassetpricing as oap

openap = oap.OpenAP()

# see what signals are available
print("Downloading signal docs...")
docs = openap.dl_signal_doc('pandas')
print(docs.head(20))

# download a few specific signals
print("\nDownloading signals...")
data = openap.dl_signal('pandas', ['BM', 'Mom12m', 'GP', 'Turnover'])

print(data.head())
print("\nShape:", data.shape)
print("\nColumns:", data.columns.tolist())

# save it
data.to_csv("data/processed/sample_signals.csv", index=False)
print("\nSaved to data/processed/sample_signals.csv")