"""
Threndia Sync Script.
Bidirectional synchronization with the Threndia repository.

Usage:
    python sync_threndia.py --push   (Fazenda -> Threndia)
    python sync_threndia.py --pull   (Threndia -> Fazenda)
"""
import argparse
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

THRENDIA_REMOTE = "threndia"
BRANCH_NAME = "threndia-pairing"

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
        logger.error(f"Git command failed: {' '.join(args)}\nError: {e.stderr}")
        raise

def push_to_threndia():
    """Push local threndia-pairing branch to Threndia remote."""
    logger.info(">>> PUSHING TO THRENDIA <<<")

    run_git_cmd(["fetch", THRENDIA_REMOTE], check=False)

    try:
        run_git_cmd(["push", THRENDIA_REMOTE, f"{BRANCH_NAME}:main"])
        logger.info("Successfully pushed to Threndia!")
    except Exception as e:
        logger.error(f"Push failed: {e}")

def pull_from_threndia():
    """Pull changes from Threndia to local."""
    logger.info(">>> PULLING FROM THRENDIA <<<")

    run_git_cmd(["fetch", THRENDIA_REMOTE])

    try:
        run_git_cmd(["pull", "--rebase", THRENDIA_REMOTE, "main"])
        logger.info("Successfully pulled from Threndia!")
    except Exception as e:
        logger.error(f"Pull failed: {e}")
        logger.warning("Resolve conflicts manually if needed.")

def main():
    parser = argparse.ArgumentParser(description="Sync with Threndia")
    parser.add_argument("--push", action="store_true", help="Push to Threndia")
    parser.add_argument("--pull", action="store_true", help="Pull from Threndia")

    args = parser.parse_args()

    if not (args.push or args.pull):
        print("Specify --push or --pull")
        parser.print_help()
        return

    if args.pull:
        pull_from_threndia()
    if args.push:
        push_to_threndia()

if __name__ == "__main__":
    main()
