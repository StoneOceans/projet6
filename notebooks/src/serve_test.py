import requests
import pandas as pd

# Exemple après lancement :
# mlflow models serve -m runs:/ID_DU_RUN/model -p 5001 --no-conda

sample = pd.read_csv('../data/processed/dataset_model.csv').drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore').head(1)

payload = {
    'dataframe_split': sample.to_dict(orient='split')
}

response = requests.post('http://127.0.0.1:5001/invocations', json=payload)
print(response.text)
