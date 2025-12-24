#!/bin/bash
set -euxo pipefail

# MASTER-ONLY guard (hostnames typically end with -m on Dataproc master)
HOSTNAME="$(hostname)"
if [[ "${HOSTNAME}" != *-m ]]; then
  echo "Not master node (${HOSTNAME}). Skipping slackbot."
  exit 0
fi

BOT_DIR="/opt/slackbot"
CODE_ZIP_GCS="gs://dataproc-data-analytics-cluster-bucket-1/Anurag_chauhan/slackbot_dataproc_clean final.zip"   # <-- change me


apt-get update
apt-get install -y python3-venv python3-pip unzip

mkdir -p "${BOT_DIR}"
cd "${BOT_DIR}"

gsutil cp "${CODE_ZIP_GCS}" /tmp/slackbot.zip
unzip -o /tmp/slackbot.zip -d "${BOT_DIR}"

python3 -m venv "${BOT_DIR}/.venv"
"${BOT_DIR}/.venv/bin/pip" install -r "${BOT_DIR}/requirements.txt"

# Put secrets here for now (upgrade later to Secret Manager)
cat > /etc/slackbot.env << 'EOF'
SLACK_BOT_TOKEN=xoxb-REPLACE_ME
SLACK_APP_TOKEN=xapp-REPLACE_ME
GEMINI_API_KEY=REPLACE_ME
GEMINI_MODEL=gemini-1.5-pro
LOG_LEVEL=INFO
PREVIEW_ROWS=20
CSV_MAX_ROWS=5000
SQL_SCHEMA_PROMPT_PATH=/opt/slackbot/schema_prompt.txt
EOF
chmod 600 /etc/slackbot.env

cat > /etc/systemd/system/slackbot.service << EOF
[Unit]
Description=Slack Bot (Socket Mode) on Dataproc Master
After=network.target

[Service]
Type=simple
WorkingDirectory=${BOT_DIR}
EnvironmentFile=/etc/slackbot.env
ExecStart=${BOT_DIR}/.venv/bin/python ${BOT_DIR}/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable slackbot
systemctl restart slackbot
systemctl --no-pager status slackbot || true
