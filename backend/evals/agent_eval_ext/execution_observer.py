"""One journal per process; identities join host, adapter and case traces without shared writes."""

import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

from research_engine.eval_trace import Span, active_trace, eval_trace_run, trace_span
from research_engine.trace_content import SECRET_NAME


def environment_secrets():
    return tuple(v for k, v in os.environ.items() if SECRET_NAME.search(k) and len(v) >= 4)


def _drain_pipe(read_fd, stop, tail):
    size, remaining, empty_after_stop = 0, 64, False
    while not stop.is_set() or remaining > 0:
        try:
            chunk = os.read(read_fd, 4096)
        except BlockingIOError:
            if stop.is_set():
                # EAGAIN may have happened BEFORE the last write and stop signal.
                # Retry after observing stop before declaring the pipe drained.
                if empty_after_stop:
                    break
                empty_after_stop = True
                continue
            stop.wait(0.01)
            continue
        except OSError:
            break
        if not chunk:
            break
        size += len(chunk)
        tail.extend(chunk)
        del tail[:-16000]
        if stop.is_set():
            remaining -= 1
    return size


@contextmanager
def subprocess_output(name):
    if active_trace() is None:
        yield subprocess.DEVNULL, Span(None)
        return
    # Drain into a bounded memory tail. Raw process output never lands on disk;
    # the child cannot deadlock on a full pipe while the parent waits for exit.
    try:
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
    except OSError:
        yield subprocess.DEVNULL, Span(None)
        return
    stop = threading.Event()
    tail = bytearray()
    size = 0

    def drain():
        nonlocal size
        size = _drain_pipe(read_fd, stop, tail)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    try:
        with trace_span(name, kind="subprocess") as span:
            try:
                yield write_fd, span
            finally:
                os.close(write_fd)
                stop.set()
                # Nonblocking reads plus the finite post-stop drain make this
                # join bounded by local work; never emit before the tail is ready.
                thread.join()
                text = tail.decode("utf8", errors="replace")
                if size > len(tail):
                    # Discard a cut first line; otherwise a credential whose prefix
                    # fell outside the memory tail could evade whole-value redaction.
                    text = text.partition("\n")[2] or "[oversized single log line omitted]"
                span.observe(
                    output={
                        "log_tail": text,
                        "log_bytes": size,
                        "log_tail_truncated": size > 16000,
                    }
                )
    finally:
        os.close(read_fd)


@contextmanager
def execution_scope(directory: Path, run_id: str, name: str, *, inputs=None):
    with eval_trace_run(
        directory=directory / "execution-traces",
        run_id=run_id,
        case_id="evaluation",
        enabled=True,
        capture_content=True,
        redaction_secrets=environment_secrets(),
    ):
        with trace_span(name, kind="evaluation_stage", inputs=inputs) as span:
            yield span
