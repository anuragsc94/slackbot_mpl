import os
import logging
import threading
import time

from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.slack_app import build_slack_app
from app.bot_config import load_bot_profiles


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

    bot_config_path = os.getenv("BOT_CONFIG_PATH")

    # --------------------------------------------------------------
    # Multi-bot mode: one process serves multiple Slack apps
    # Each bot can be mapped to a different BigQuery project.
    # --------------------------------------------------------------
    if bot_config_path:
        profiles = load_bot_profiles(bot_config_path)
        if not profiles:
            raise RuntimeError(f"No bots found in BOT_CONFIG_PATH={bot_config_path}")

        logger.info(f"✅ Loaded {len(profiles)} bot profile(s) from {bot_config_path}")

        threads: list[threading.Thread] = []
        for p in profiles:
            app = build_slack_app(
                p.slack_bot_token,
                bq_project_id=p.bq_project_id,
                bot_name=p.name,
            )

            def _runner(profile=p, slack_app=app) -> None:
                logger.info(
                    f"✅ Starting Slack bot `{profile.name}` (Socket Mode), "
                    f"bq_project_id={profile.bq_project_id or '(default)'}"
                )
                SocketModeHandler(slack_app, profile.slack_app_token).start()

            t = threading.Thread(target=_runner, name=f"socketmode-{p.name}", daemon=True)
            t.start()
            threads.append(t)

        # Keep main thread alive while bot threads run.
        # SocketModeHandler.start() is blocking, so each bot runs on its own thread.
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("🛑 Received KeyboardInterrupt. Exiting...")
        return

    # --------------------------------------------------------------
    # Single-bot mode: backwards compatible env vars
    # --------------------------------------------------------------
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")  # xoxb-...
    slack_app_token = os.getenv("SLACK_APP_TOKEN")  # xapp-... (Socket Mode)
    bq_project_id = os.getenv("BQ_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")

    if not slack_bot_token:
        raise RuntimeError("Missing SLACK_BOT_TOKEN in environment")

    if not slack_app_token:
        raise RuntimeError("Missing SLACK_APP_TOKEN in environment")

    app = build_slack_app(slack_bot_token, bq_project_id=bq_project_id)

    logger.info(f"✅ Starting Slack bot (Socket Mode), bq_project_id={bq_project_id or '(default)'}...")
    SocketModeHandler(app, slack_app_token).start()


if __name__ == "__main__":
    main()
