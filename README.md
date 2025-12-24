# Slackbot on Dataproc (Socket Mode)

This folder is a cleaned, deployment-ready version of your notebook.

## What changed vs notebook
- **No secrets in code**: tokens/keys come from environment variables.
- Notebook-only commands (`!pip install ...`) removed.
- Schema / long prompt moved to `schema_prompt.txt`.
- One entrypoint: `main.py`.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill values
export $(cat .env | xargs)
python main.py
```

## Dataproc deployment idea
- Zip this folder and upload to GCS
- Use a Dataproc init-action script to install deps and start `main.py` via systemd on the **master** node.

See `scripts/dataproc_init_action.sh` for a template.
