# Dubai_real_state_dashboard
Monitor transaction for Aapartments in Dubai

Production:

- store production credentials in the deployment secret store, not in git
- for local development, copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
- test the last 4 months of Dubai Data API records with `python smoke_test_dda_api.py --limit 100000 --require-records`

Required secret keys:

- `DDA_BASE_URL`
- `DDA_SECURITY_APPLICATION_IDENTIFIER`
- `DDA_CLIENT_ID`
- `DDA_CLIENT_SECRET`
- `DDA_ENTITY`
- `DDA_DATASET`
- `DDA_VERIFY_SSL`

Next Steps:

- refactor
- add map chart of price changes
- scrape real state website
