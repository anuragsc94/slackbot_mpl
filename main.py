import os
import logging

from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.slack_app import build_slack_app

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("slackbot")

def main() -> None:
    slack_bot_token = os.environ["SLACK_BOT_TOKEN"]          # xoxb-...
    slack_app_token = os.environ["SLACK_APP_TOKEN"]          # xapp-... (Socket Mode)

    app = build_slack_app(slack_bot_token)

    logger.info("✅ Starting Slack bot (Socket Mode)...")
    SocketModeHandler(app, slack_app_token).start()

if __name__ == "__main__":
    main()
