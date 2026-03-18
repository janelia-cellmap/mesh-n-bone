import os
import sys
import time
import socket
import smtplib
import getpass
import logging
import traceback
from email.mime.text import MIMEText
from contextlib import ContextDecorator, contextmanager
from subprocess import Popen, PIPE, TimeoutExpired, run as subprocess_run
from datetime import datetime

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class Timing_Messager(ContextDecorator):
    """Context manager class to time operations"""

    def __init__(self, base_message, logger):
        self._base_message = base_message
        self._logger = logger

    def __enter__(self):
        print_with_datetime(f"{self._base_message}...", self._logger)
        self._start_time = time.time()
        return self

    def __exit__(self, *exc):
        print_with_datetime(
            f"{self._base_message} completed in {time.time()-self._start_time}!",
            self._logger,
        )
        return False


def print_with_datetime(output, logger):
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"{now}: {output}")


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """Context manager that redirects a file object or file descriptor to a new
    file descriptor, including C/C++ output.

    Lifted from https://stackoverflow.com/a/22434262/162094 (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            yield stdout
            return
    except ValueError:
        pass

    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        flush(stdout)
        try:
            os.dup2(fileno(to), stdout_fd)
        except ValueError:
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)
        try:
            yield stdout
        finally:
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)


def flush(stream):
    try:
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def tee_streams(output_path, append=False):
    """Context manager. All stdout and stderr will be tee'd to a file on disk."""
    if append:
        append = "-a"
    else:
        append = ""

    tee = Popen(
        f"tee {append} {output_path}",
        shell=True,
        stdin=PIPE,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setpgrp,
    )
    try:
        try:
            with stdout_redirected(tee.stdin, stdout=sys.stdout):
                with stdout_redirected(sys.stdout, stdout=sys.stderr):
                    yield
        finally:
            tee.stdin.close()
            try:
                tee.wait(1.0)
            except TimeoutExpired:
                pass
    except:
        with open(output_path, "a") as f:
            traceback.print_exc(file=f)
        raise


@contextmanager
def email_on_exit(email_config, workflow_name, execution_dir, logpath):
    """Context manager. Sends an email when the context exits with success/fail status."""
    if not email_config["send"]:
        yield
        return

    if not email_config["addresses"]:
        logger.warning(
            "Your config enabled the exit-email feature, but "
            "no email addresses were listed. Nothing will be sent."
        )
        yield
        return

    user = getpass.getuser()
    host = socket.gethostname()
    jobname = os.environ.get("LSB_JOBNAME", None)

    addresses = []
    for address in email_config["addresses"]:
        if address == "JANELIA_USER":
            address = f"{user}@janelia.hhmi.org"
        addresses.append(address)

    start_time = time.time()

    def send_email(headline, result, error_str=None):
        elapsed = time.time() - start_time
        body = headline + f"Duration: {elapsed:.1f}s\nExecution directory: {execution_dir}\n"

        if jobname:
            body += f"Job name: {jobname}\n"
        if error_str:
            body += f"Error: {error_str}\n"

        if email_config["include-log"]:
            try:
                subprocess_run("sync", timeout=10.0)
                time.sleep(2.0)
            except TimeoutExpired:
                logger.warning("Timed out while waiting for filesystem sync")

            body += "\nLOG (possibly truncated):\n\n"
            with open(f"{logpath}", "r") as log:
                body += log.read()

        msg = MIMEText(body)
        msg["Subject"] = f"Workflow exited: {result}"
        msg["From"] = f"mesh-n-bone <{user}@{host}>"
        msg["To"] = ",".join(addresses)

        try:
            s = smtplib.SMTP("mail.hhmi.org")
            s.sendmail(msg["From"], addresses, msg.as_string())
            s.quit()
        except Exception:
            logger.error(
                "Failed to send completion email. Perhaps your machine "
                "is not configured to send login-less email."
            )

    try:
        yield
    except BaseException as ex:
        send_email(
            f"Workflow {workflow_name} failed: {type(ex)}\n", "FAILED", str(ex)
        )
        raise
    else:
        send_email(f"Workflow {workflow_name} exited successfully.\n", "success")


def capture_draco_output(fd, fn, *args, **kwargs):
    """Capture output from a file descriptor during a function call.
    Used to detect Draco encoding errors written to stderr."""
    orig_fd = os.dup(fd)
    r, w = os.pipe()
    os.dup2(w, fd)
    os.close(w)

    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        result = e
    finally:
        os.dup2(orig_fd, fd)
        os.close(orig_fd)

    captured = os.read(r, 10_000_000).decode(errors="ignore")
    os.close(r)
    if "error" in captured:
        raise Exception(f"Draco error: {captured}")
    return result, captured
