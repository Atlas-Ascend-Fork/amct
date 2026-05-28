import os

from loguru import logger


def ensure_log_dir(args):
    log_dir = getattr(args, "log_dir", "") or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    return log_dir


def setup_run_logging(args, task_name: str):
    log_dir = ensure_log_dir(args)
    log_path = os.path.join(log_dir, f"{task_name}.log")
    sink_id = logger.add(
        log_path,
        level="INFO",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    return sink_id, log_path
