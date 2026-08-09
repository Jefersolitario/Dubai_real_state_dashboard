# Data Dubai Production Access

Production access is configured through the DDA iPaaS OAuth flow.

## Local Configuration

Create `.streamlit/secrets.toml` from `.streamlit/secrets.example.toml` and keep it out of git.

```toml
DDA_BASE_URL = "https://apis.data.dubai"
DDA_SECURITY_APPLICATION_IDENTIFIER = "paste-security-application-identifier-here"
DDA_CLIENT_ID = "paste-client-id-here"
DDA_CLIENT_SECRET = "paste-client-secret-here"
DDA_ENTITY = "dld"
DDA_DATASET = "dld_transactions-open-api"
DDA_VERIFY_SSL = true
```

## Production Secret Storage

Do not commit `.streamlit/secrets.toml` or put credentials in `dubai_dashboard.py`.
Use the secret manager for the deployment target:

- Streamlit Community Cloud: app settings -> Secrets, paste the same TOML keys.
- Docker or VM: inject the same values as environment variables.
- GitHub Actions deploy: store them as GitHub Actions secrets and pass them to the hosting platform.
- Cloud platforms: use the native secret manager, such as AWS Secrets Manager,
  Azure Key Vault, Google Secret Manager, or the platform's app secret settings.

Only `.streamlit/secrets.example.toml` should be committed.

## Validation Commands

Quick connectivity and mapping check:

```powershell
.\.venv\Scripts\python.exe .\smoke_test_dda_api.py --limit 10 --require-records
```

Full recent-window check for the dashboard:

```powershell
.\.venv\Scripts\python.exe .\smoke_test_dda_api.py --limit 100000 --require-records
```

By default the smoke test queries flat transactions for the latest 4 months,
ordered by `instance_date` descending.

## Production Notes

- Production base URL: `https://apis.data.dubai`
- Dataset endpoint: `/secure/ddads/openapi/1.0.0/dld/dld_transactions-open-api`
- The secure token endpoint may reject this app in production, but the standard
  OAuth client credentials endpoint succeeds and is used as a fallback.
- Do not commit or paste `client_secret`, `client_id`, or
  `x-DDA-SecurityApplicationIdentifier` into tracked files.
