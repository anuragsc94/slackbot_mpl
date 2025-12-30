import os
import logging

from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.slack_app import build_slack_app


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("slackbot")


# ------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------

def main() -> None:
    # Load environment variables from .env
    load_dotenv()

    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")    # xoxb-...
    slack_app_token = os.getenv("SLACK_APP_TOKEN")    # xapp-... (Socket Mode)

    if not slack_bot_token:
        raise RuntimeError("Missing SLACK_BOT_TOKEN in environment")

    if not slack_app_token:
        raise RuntimeError("Missing SLACK_APP_TOKEN in environment")

    app = build_slack_app(slack_bot_token)

    logger.info("✅ Starting Slack bot (Socket Mode)...")
    SocketModeHandler(app, slack_app_token).start()


if __name__ == "__main__":
    main()
