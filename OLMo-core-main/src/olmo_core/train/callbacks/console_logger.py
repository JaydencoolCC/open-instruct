import logging
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Dict, List, Optional

from tqdm import tqdm

from olmo_core.utils import format_float, format_timedelta

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ConsoleLoggerCallback(Callback):
    """
    Logs progress and a subset of metrics to the console.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    log_interval: int = 1
    """
    How often, in steps, to log progress to the console.
    """

    metrics_log_interval: Optional[int] = None
    """
    How often, in steps, to log metrics to the console. If not set, defaults to :data:`log_interval`.
    """

    metrics: List[str] = field(
        default_factory=lambda: [
            "train/CE loss",
            "train/PPL",
            "train/Z loss",
            "train/load balancing loss",
            "train/router Z loss",
            "train/block */load imbalance",
            "gpu_memory/*",
            "optim/total grad norm",
            "optim/step skipped",
            "optim/LR*",
            "throughput/*",
            "checkpoint/*",
        ]
    )
    """
    Metrics to log to the console. Wildcards are supported.
    """

    use_tqdm: bool = False
    """
    Use a tqdm progress bar instead of multi-line log messages.
    """

    _progress_bar = None

    def pre_train(self):
        if self.use_tqdm and sys.stdout.isatty():
            self._progress_bar = tqdm(
                total=self.trainer.max_steps,
                initial=self.step,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
                file=sys.stdout,
            )

    def post_step(self):
        if self.use_tqdm:
            if self._progress_bar is not None:
                self._update_progress_bar(self.step)
            return

        if self._should_log_metrics(self.step):
            # Will log to console from `self.log_metrics()`.
            return

        if self.step % self.log_interval != 0:
            return

        log.info(self._get_progress_marker(self.step))

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self._should_log_metrics(step):
            return

        if self.use_tqdm:
            if self._progress_bar is not None:
                self._update_progress_bar(step)
                self._progress_bar.set_postfix_str(self._format_compact_metrics(metrics))
            else:
                log.info(
                    f"{self._get_progress_marker(step, include_eta=True)} "
                    f"{self._format_compact_metrics(metrics)}"
                )
            return

        prefix = self._get_progress_marker(step, include_eta=True)
        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if any(fnmatch(name, pat) for pat in self.metrics)
                ]
            )
        )

    def _get_progress_marker(self, step: int, include_eta: bool = False) -> str:
        if include_eta and (eta := self.trainer.training_progress.time_remaining) is not None:
            eta_str = format_timedelta(eta).replace(", ", "")
            if self.trainer.hard_stop:
                eta_str = f"{eta_str}(hard stop)"
            return f"[step={step}/{self.trainer.max_steps or '???'},epoch={self.trainer.epoch},eta={eta_str}]"
        else:
            return f"[step={step}/{self.trainer.max_steps or '???'},epoch={self.trainer.epoch}]"

    def _should_log_metrics(self, step: int) -> bool:
        metrics_log_interval = self.metrics_log_interval or self.log_interval
        if step == 1 or (step > 1 and step % metrics_log_interval == 0):
            return True
        else:
            return False

    def _update_progress_bar(self, step: int):
        if self._progress_bar is None:
            return

        delta = step - self._progress_bar.n
        if delta > 0:
            self._progress_bar.update(delta)

    def _format_compact_metrics(self, metrics: Dict[str, float]) -> str:
        parts = [f"epoch={self.trainer.epoch}"]

        if (value := metrics.get("train/CE loss")) is not None:
            parts.append(f"loss={format_float(value)}")
        if (value := metrics.get("train/PPL")) is not None:
            parts.append(f"ppl={format_float(value)}")
        if (value := metrics.get("optim/LR (group 0)")) is not None:
            parts.append(f"lr={format_float(value)}")
        if (value := metrics.get("gpu_memory/GPU active mem (GiB)")) is not None:
            parts.append(f"mem={format_float(value)}GiB")
        if (value := metrics.get("throughput/device/BPS (actual avg)")) is not None:
            parts.append(f"bps={format_float(value)}")
        if (value := metrics.get("throughput/device/TPS (actual avg)")) is not None:
            parts.append(f"tps={format_float(value)}")
        if (value := metrics.get("throughput/device/MFU (actual avg)")) is not None:
            parts.append(f"mfu={format_float(value)}%")
        if (value := metrics.get("optim/total grad norm")) is not None:
            parts.append(f"grad={format_float(value)}")

        return " ".join(parts)

    def close(self):
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
