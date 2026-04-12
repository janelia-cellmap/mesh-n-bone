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
    """Context manager and decorator that logs elapsed wall-clock time.

    Prints a "starting..." message on entry and a "completed in <seconds>!"
    message on exit.

    Parameters
    ----------
    base_message : str
        Human-readable label for the operation being timed.
    logger : logging.Logger
        Logger instance used for output.
    """

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
    """Log a message prefixed with the current date and time.

    Parameters
    ----------
    output : str
        Message to log.
    logger : logging.Logger
        Logger instance used for output (at INFO level).
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"{now}: {output}")


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """Redirect a file descriptor (including C/C++ level output) to another target.

    This works at the OS file-descriptor level, so it captures output from
    native extensions that bypass Python's ``sys.stdout``.

    Adapted from https://stackoverflow.com/a/22434262/162094 (MIT License).

    Parameters
    ----------
    to : str or file-like, optional
        Destination file path or file object.  Defaults to ``os.devnull``.
    stdout : file-like or None, optional
        The stream to redirect.  Defaults to ``sys.stdout``.

    Yields
    ------
    file-like
        The original *stdout* object (useful for nested redirections).
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
    """Duplicate stdout and stderr to a file on disk (like Unix ``tee``).

    Both Python-level and C-level output is captured by redirecting the
    underlying file descriptors through a ``tee`` subprocess.

    Parameters
    ----------
    output_path : str
        Path to the log file.
    append : bool, optional
        If ``True``, append to the file instead of overwriting.
        Default is ``False``.
    """
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
    """Send a notification email when the enclosed block exits.

    On success an informational email is sent; on failure the exception
    details are included. If ``email_config["send"]`` is ``False`` or no
    addresses are configured, the context manager is a no-op.

    Parameters
    ----------
    email_config : dict
        Configuration dictionary with keys ``"send"`` (bool),
        ``"addresses"`` (list of str), and ``"include-log"`` (bool).
        The special address ``"JANELIA_USER"`` is expanded to
        ``<unix-user>@janelia.hhmi.org``.
    workflow_name : str
        Human-readable workflow name used in the email subject.
    execution_dir : str
        Path to the execution directory, included in the email body.
    logpath : str
        Path to the log file.  Appended to the email body when
        ``email_config["include-log"]`` is ``True``.
    """
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
    """Call a function while capturing output from a file descriptor.

    Temporarily redirects *fd* (typically ``sys.stderr.fileno()``) to a
    pipe, runs *fn*, then reads the captured bytes. If the captured text
    contains the word ``"error"``, an exception is raised.

    Parameters
    ----------
    fd : int
        OS-level file descriptor to capture (e.g. ``2`` for stderr).
    fn : callable
        Function to invoke.
    *args
        Positional arguments forwarded to *fn*.
    **kwargs
        Keyword arguments forwarded to *fn*.

    Returns
    -------
    result : object
        Return value of ``fn(*args, **kwargs)``.
    captured : str
        Text captured from the file descriptor.

    Raises
    ------
    Exception
        If the captured output contains the substring ``"error"``.
    """
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
