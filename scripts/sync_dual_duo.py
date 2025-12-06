"""
Dual Duo Sync Script.
Bidirectional synchronization between Fazenda and Threndia (pairduo branch).

Usage:
    python sync_dual_duo.py --push   (Fazenda -> Threndia)
    python sync_dual_duo.py --pull   (Threndia -> Fazenda)
"""
import argparse
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

THRENDIA_REMOTE = "threndia"
BRANCH_NAME = "pairduo"

def run_git_cmd(args: list, check: bool = True):
    try:
        logger.debug(f"Running: git {' '.join(args)}")
        result = subprocess.run(
            ["git"] + args,
            check=check,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {' '.join(args)}\nInclude: {e.stderr}")
        raise

def push_to_threndia():
    """Push local pairduo branch to Threndia remote."""
    logger.info(">>> STARTING PUSH (Fazenda -> Threndia) <<<")

    # 1. Fetch remote to ensure we have refs
    logger.info("Fetching remote...")
    run_git_cmd(["fetch", THRENDIA_REMOTE], check=False)

    # 2. Push
    logger.info(f"Pushing {BRANCH_NAME} to {THRENDIA_REMOTE}...")
    try:
        # Pushing to both main (if it's the target) and pairduo to be safe/redundant as per prev plan
        # User requested integration, assuming direct branch mapping pairduo:pairduo is safest
        # unless Threndia base IS pairduo code.
        # Let's push to pairduo on remote.
        run_git_cmd(["push", THRENDIA_REMOTE, f"{BRANCH_NAME}:{BRANCH_NAME}"])
        logger.info("Successfully pushed to Threndia/pairduo.")
    except Exception as e:
        logger.error(f"Push failed: {e}")

def pull_from_threndia():
    """Pull changes from Threndia to local pairduo."""
    logger.info(">>> STARTING PULL (Threndia -> Fazenda) <<<")

    # 1. Fetch
    logger.info("Fetching remote...")
    run_git_cmd(["fetch", THRENDIA_REMOTE])

    # 2. Pull (rebase to keep history clean)
    logger.info(f"Pulling from {THRENDIA_REMOTE}/{BRANCH_NAME}...")
    try:
        run_git_cmd(["pull", "--rebase", THRENDIA_REMOTE, BRANCH_NAME])
        logger.info("Successfully pulled from Threndia.")
    except Exception as e:
        logger.error(f"Pull failed (conflict likely): {e}")
        logger.warning("Please resolve conflicts manually.")

def main():
    parser = argparse.ArgumentParser(description="Sync Fazenda and Threndia")
    parser.add_argument("--push", action="store_true", help="Push changes to Threndia")
    parser.add_argument("--pull", action="store_true", help="Pull changes from Threndia")

    args = parser.parse_args()

    if not (args.push or args.pull):
        print("Please specify --push or --pull")
        parser.print_help()
        return

    try:
        if args.pull:
            pull_from_threndia()
        if args.push:
            push_to_threndia()

    except Exception as e:
        logger.error(f"Sync operation aborted: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
