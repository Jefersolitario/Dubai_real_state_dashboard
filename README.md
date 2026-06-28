# Dubai Real Estate Dashboard

Streamlit dashboard for Dubai apartment transactions. The production app reads
normalized DLD transaction data from a Google Cloud Storage Parquet snapshot, so
the dashboard can start quickly without querying the Dubai Data API on every
cold start.

## Run Locally

```powershell
pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dubai_dashboard.py
```

## Data Flow

1. `store_dld_transactions_gcs.py` fetches recent DLD transactions from the
   Dubai Data API.
2. The script normalizes the records and writes:

```text
gs://dubai-real-estate-dashboard-jef/dld_transactions/dld_transactions_latest.parquet
```

3. `dubai_dashboard.py` loads that GCS Parquet snapshot first.
4. If the GCS snapshot is unavailable, the dashboard falls back to the DDA API.

Current verified snapshot:

```text
rows: 48,028
columns: 69
date coverage: 2026-03-02 to 2026-06-25
```

## Streamlit Secrets

Keep secrets in `.streamlit/secrets.toml` locally and in Streamlit Cloud app
settings for deployment. Do not commit secrets.

Minimum GCS secrets:

```toml
GCS_BUCKET = "dubai-real-estate-dashboard-jef"
GCS_TRANSACTIONS_OBJECT = "dld_transactions/dld_transactions_latest.parquet"

GCP_SERVICE_ACCOUNT_JSON = '''
{
  "paste": "the full downloaded service account JSON here"
}
'''
```

DDA secrets are needed for refreshing the snapshot and for API fallback:

```toml
DDA_BASE_URL = "https://apis.data.dubai"
DDA_SECURITY_APPLICATION_IDENTIFIER = "..."
DDA_CLIENT_ID = "..."
DDA_CLIENT_SECRET = "..."
DDA_ENTITY = "dld"
DDA_DATASET = "dld_transactions-open-api"
DDA_VERIFY_SSL = true
```

## GCS Setup

Current resources:

```text
Project: realstateproject
Bucket: dubai-real-estate-dashboard-jef
Service account: dubai-dashboard-gcs@realstateproject-500813.iam.gserviceaccount.com
Bucket role: Cloud Storage -> Storage Object Admin
```

Bucket settings:

```text
Public access prevention: enabled
Access control: uniform
Storage class: standard
Location: EU multi-region
```

## Maintenance Commands

Test GCS read/write/delete:

```powershell
.\.venv\Scripts\python.exe .\smoke_test_gcs.py
```

Refresh the production Parquet snapshot:

```powershell
.\.venv\Scripts\python.exe .\store_dld_transactions_gcs.py
```

Test DDA connectivity:

```powershell
.\.venv\Scripts\python.exe .\smoke_test_dda_api.py --limit 10 --require-records
```

## Commit Safety

Commit code and docs only. Never commit:

```text
.streamlit/secrets.toml
*.json service-account keys
```
