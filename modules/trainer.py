import os
import re
import signal
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex patterns for parsing actual diffusion-pipe stdout
# "Started new epoch: 5"
EPOCH_PATTERN = re.compile(r"Started new epoch:\s*(\d+)")
# DeepSpeed loss logging: "loss=0.1234" or "loss: 0.1234" or "loss 0.1234"
LOSS_PATTERN = re.compile(r"\bloss[=:\s]+([0-9]+\.?[0-9]*(?:[eE][+\-]?\d+)?)")
# "Saving model to directory epoch50"
SAVING_PATTERN = re.compile(r"Saving model to directory\s+(\S+)")
# "TRAINING COMPLETE!"
COMPLETE_PATTERN = re.compile(r"TRAINING COMPLETE!")

# Progress range for training phase (30-90% of total job)
TRAIN_PROGRESS_START = 30
TRAIN_PROGRESS_END = 90


class TrainingRunner:
    def __init__(self, training_toml, work_dir, total_epochs=None, diffusion_pipe_dir="/diffusion_pipe"):
        self.training_toml = training_toml
        self.work_dir = work_dir
        self.total_epochs = total_epochs
        self.diffusion_pipe_dir = diffusion_pipe_dir
        self.process = None
        self._terminated = False

    def run(self):
        """
        Generator that runs deepspeed training and yields progress updates.

        Yields dicts: {"progress": 0-100, "message": str, "epoch": int, "loss": float}

        Parses actual diffusion-pipe stdout:
          "Started new epoch: 5"
          "Saving model to directory epoch50"
          "TRAINING COMPLETE!"
        """
        train_script = os.path.join(self.diffusion_pipe_dir, "train.py")

        if not os.path.exists(train_script):
            raise FileNotFoundError(
                f"train.py not found at {train_script}. "
                "Ensure diffusion-pipe is installed."
            )

        cmd = [
            "deepspeed",
            "--num_gpus=1",
            train_script,
            "--deepspeed",
            "--config", self.training_toml,
        ]

        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"

        logger.info(f"Starting training: {' '.join(cmd)}")
        yield {"progress": TRAIN_PROGRESS_START, "message": "Starting training..."}

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.diffusion_pipe_dir,
        )

        current_epoch = 0
        last_loss = None

        try:
            for line in self.process.stdout:
                line = line.rstrip()
                if not line:
                    continue

                logger.debug(line)

                # "Started new epoch: 5"
                epoch_match = EPOCH_PATTERN.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    progress = _epoch_to_progress(current_epoch, self.total_epochs)
                    msg = f"Epoch {current_epoch}"
                    if self.total_epochs:
                        msg += f"/{self.total_epochs}"
                    if last_loss is not None:
                        msg += f" | loss: {last_loss:.6f}"
                    yield {
                        "progress": progress,
                        "message": msg,
                        "epoch": current_epoch,
                        "loss": last_loss,
                    }
                    continue

                # DeepSpeed loss output (e.g. "loss=0.1234")
                loss_match = LOSS_PATTERN.search(line)
                if loss_match:
                    try:
                        last_loss = float(loss_match.group(1))
                    except ValueError:
                        pass

                # "Saving model to directory epoch50"
                save_match = SAVING_PATTERN.search(line)
                if save_match:
                    saved_name = save_match.group(1)
                    yield {
                        "progress": _epoch_to_progress(current_epoch, self.total_epochs),
                        "message": f"Saved model: {saved_name}",
                        "epoch": current_epoch,
                        "loss": last_loss,
                    }
                    continue

                # "TRAINING COMPLETE!"
                if COMPLETE_PATTERN.search(line):
                    logger.info("Training complete signal received")

        except Exception as e:
            logger.error(f"Error reading training output: {e}")
            raise

        self.process.wait()
        exit_code = self.process.returncode

        if self._terminated:
            raise RuntimeError("Training was terminated by signal")

        if exit_code != 0:
            raise RuntimeError(f"Training failed with exit code {exit_code}")

        yield {
            "progress": TRAIN_PROGRESS_END,
            "message": f"Training complete. Final epoch: {current_epoch}",
            "epoch": current_epoch,
            "loss": last_loss,
        }

    def terminate(self):
        """Gracefully terminate the training process."""
        self._terminated = True
        if self.process and self.process.poll() is None:
            logger.info("Sending SIGTERM to training process...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Training process didn't stop, sending SIGKILL")
                self.process.kill()


def find_latest_checkpoint(output_dir):
    """Find the latest saved LoRA checkpoint directory.

    diffusion-pipe creates a timestamped run dir inside output_dir:
      output_dir/20240101_12-00-00/epoch10/
      output_dir/20240101_12-00-00/epoch20/
      output_dir/20240101_12-00-00/step100/

    Saves are named 'epoch{N}' or 'step{N}'. We look for the highest
    epoch first, then step as fallback.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    def _find_model_dirs(search_dir):
        """Find epoch{N} and step{N} dirs, return (number, path) list."""
        found = []
        for d in search_dir.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            # Skip deepspeed checkpoint dirs (global_step{N})
            if name.startswith("global_step"):
                continue
            if name.startswith("epoch"):
                try:
                    found.append((int(name[5:]), d))
                except ValueError:
                    continue
            elif name.startswith("step"):
                try:
                    found.append((int(name[4:]), d))
                except ValueError:
                    continue
        return found

    # diffusion-pipe always creates a timestamped run subdir
    # Check run subdirectories (most recent first by name = chronological)
    model_dirs = []
    subdirs = sorted(
        [d for d in output_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for run_dir in subdirs:
        model_dirs = _find_model_dirs(run_dir)
        if model_dirs:
            break

    # Fallback: check directly in output_dir (shouldn't happen but safe)
    if not model_dirs:
        model_dirs = _find_model_dirs(output_path)

    if not model_dirs:
        return None

    model_dirs.sort(key=lambda x: x[0], reverse=True)
    latest = model_dirs[0][1]
    logger.info(f"Latest checkpoint: {latest}")
    return str(latest)


def _epoch_to_progress(current, total):
    """Map epoch progress to the 30-90% range."""
    if not total:
        return TRAIN_PROGRESS_START
    fraction = current / total
    return int(TRAIN_PROGRESS_START + fraction * (TRAIN_PROGRESS_END - TRAIN_PROGRESS_START))
