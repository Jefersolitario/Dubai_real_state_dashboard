# Dubai_real_state_dashboard
Monitor transaction for Aapartments in Dubai

Production:

- store production credentials in the deployment secret store, not in git
- for local development, copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
- test the last 4 months of Dubai Data API records with `python smoke_test_dda_api.py --limit 100000 --require-records`
- the deployed dashboard loads the GCS Parquet snapshot first, then falls back to the DDA API only if the cache cannot be read

Required secret keys:

- `DDA_BASE_URL`
- `DDA_SECURITY_APPLICATION_IDENTIFIER`
- `DDA_CLIENT_ID`
- `DDA_CLIENT_SECRET`
- `DDA_ENTITY`
- `DDA_DATASET`
- `DDA_VERIFY_SSL`

Google Cloud Storage setup:

Current GCS resources created for this project:

- Google Cloud project: `realstateproject`
- GCS bucket: `dubai-real-estate-dashboard-jef`
- Service account: `dubai-dashboard-gcs@realstateproject-500813.iam.gserviceaccount.com`
- Bucket role granted to the service account: `Cloud Storage -> Storage Object Admin`

Bucket settings:

- Public access prevention: enabled
- Access control: uniform
- Storage class: standard
- Location: EU multi-region

If recreating this setup:

1. Create/select a Google Cloud project.
2. Create a Cloud Storage bucket with a globally unique lowercase name.
3. Keep public access prevention enabled.
4. Create a service account named `dubai-dashboard-gcs`.
5. Grant that service account bucket-level `Storage Object Admin` access.
6. Create a JSON key for the service account.
7. Store the bucket name and JSON key in `.streamlit/secrets.toml`; do not commit secrets.

Local `.streamlit/secrets.toml` values:

```toml
GCS_BUCKET = "dubai-real-estate-dashboard-jef"
GCS_TEST_PREFIX = "dubai-dashboard-smoke-tests"
GCS_TRANSACTIONS_OBJECT = "dld_transactions/dld_transactions_latest.parquet"

GCP_SERVICE_ACCOUNT_JSON = '''
{
  "paste": "the full downloaded service account JSON here"
}
'''
```

Smoke test:

```powershell
.\.venv\Scripts\python.exe .\smoke_test_gcs.py
```

The script writes a tiny Parquet file, reads it back, verifies the round trip,
and deletes the test object by default. Configure credentials with
`GOOGLE_APPLICATION_CREDENTIALS`, Google application-default credentials, or
`GCP_SERVICE_ACCOUNT_JSON`.

Store/update the real DLD transaction snapshot:

```powershell
.\.venv\Scripts\python.exe .\store_dld_transactions_gcs.py
```

The dashboard reads `GCS_TRANSACTIONS_OBJECT` from `GCS_BUCKET` on startup.
Refresh this object before deploying when you want a fresher production cache.

Next Steps:

- refactor
- add map chart of price changes
- scrape real state website
