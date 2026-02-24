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
import argparse
import yaml
from yaml.loader import SafeLoader
import numpy as np

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class Timing_Messager(ContextDecorator):
    """Context manager class to time operations"""

    def __init__(self, base_message, logger):
        """Initialize instance with a message and logger

        Args:
            base_message ('str'): Message for logger
            logger: logger to be used
        """

        self._base_message = base_message
        self._logger = logger

    def __enter__(self):
        """Set the start time and print the status message"""

        print_with_datetime(f"{self._base_message}...", self._logger)
        self._start_time = time.time()
        return self

    def __exit__(self, *exc):
        """Print the exit message and elapsed time"""

        print_with_datetime(
            f"{self._base_message} completed in {time.time()-self._start_time}!",
            self._logger,
        )
        return False


def print_with_datetime(output, logger):
    """[summary]

    Args:
        output ([type]): [description]
        logger ([type]): [description]
    """
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"{now}: {output}")


def read_run_config(config_path):
    """Reads the run config from config_path and stores them

    Args:
        config_path ('str'): Path to config directory

    Returns:
        Dicts of required_settings and optional_decimation_settings
    """

    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
        required_settings = config["required_settings"]
        optional_decimation_settings = config.get("optional_decimation_settings", {})

        if "box_size" not in optional_decimation_settings:
            optional_decimation_settings["box_size"] = None
        else:
            box_size = optional_decimation_settings["box_size"]
            if box_size is not None:
                box_size = np.atleast_1d(np.asarray(box_size, dtype=float))
                if box_size.size == 1:
                    box_size = np.full(3, box_size.item())
                elif box_size.size != 3:
                    raise ValueError(
                        f"box_size must be a scalar or a 3-element list, got {box_size}"
                    )
                optional_decimation_settings["box_size"] = box_size
        if "skip_decimation" not in optional_decimation_settings:
            optional_decimation_settings["skip_decimation"] = False
        if "decimation_factor" not in optional_decimation_settings:
            optional_decimation_settings["decimation_factor"] = 2
        if "aggressiveness" not in optional_decimation_settings:
            optional_decimation_settings["aggressiveness"] = 7
        if "delete_decimated_meshes" not in optional_decimation_settings:
            optional_decimation_settings["delete_decimated_meshes"] = False

        return required_settings, optional_decimation_settings


def parser_params():
    """Parse command line parameters including the config path and number of workers."""

    parser = argparse.ArgumentParser(
        description="Code to convert single-scale (or a set of multi-scale) meshes to the neuroglancer multi-resolution mesh format"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to directory containing run-config.yaml and dask-config.yaml",
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Number of workers to launch (i.e. each worker is launched with a single bsub command)",
    )

    return parser.parse_args()


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.

    Redirects a file object or file descriptor to a new file descriptor.

    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')

    Motivation: In pure-Python, you can redirect all print() statements like this:

        sys.stdout = open('myfile.txt')

        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.
    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.

    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError:  # filename
        pass

    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        # libc.fflush(None)  # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def tee_streams(output_path, append=False):
    """
    Context manager.
    All stdout and stderr will be tee'd to a a file on disk.
    (in addition to appearing on the original stdout streams).

    Note: Stdout and stderr will be merged, in both the tee file and the console.
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
        universal_newlines=True,  # line buffering
        preexec_fn=os.setpgrp,
    )  # Spawn the tee process in its own process group,
    # so it won't receive SIGINT.
    # (Otherwise it might close its input stream too early if the user hits Ctrl+C.)
    try:
        try:
            with stdout_redirected(tee.stdin, stdout=sys.stdout):  # pipe stdout to tee
                with stdout_redirected(
                    sys.stdout, stdout=sys.stderr
                ):  # merge stderr into stdout
                    yield
        finally:
            tee.stdin.close()
            try:
                tee.wait(1.0)
            except TimeoutExpired:
                pass
    except:
        # If an exception was raised, append the traceback to the file
        with open(output_path, "a") as f:
            traceback.print_exc(file=f)
        raise


@contextmanager
def email_on_exit(email_config, workflow_name, execution_dir, logpath):
    """
    Context manager.

    Sends an email when the context exits with success/fail status in the subject line.
    Doesn't work unless sendmail() works without a password
    (i.e. probably won't work on your laptop, will work on a Janelia cluster node).

    Args:
        email_config:
            See flyemflows.workflow.base.base_schemas.ExitEmailSchema

        workflow_name:
            Name of the workflow class to be reported in the email.

        execution_dir:
            Location of the workflow config/data files to be reported in the email.

        logpath:
            Location of the logfile whose contents will be included in
            the email if email_config["include-log"] is True.

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

    with Timer() as timer:

        def send_email(headline, result, error_str=None):
            body = (
                headline + f"Duration: {timer.timedelta}\n"
                f"Execution directory: {execution_dir}\n"
            )

            if jobname:
                body += f"Job name: {jobname}\n"

            if error_str:
                body += f"Error: {error_str}\n"

            if email_config["include-log"]:
                # Sync first, in the hope that the log will flush to disk before we read it.
                # Note:
                #    Currently raised exceptions haven't been printed yet,
                #    so they aren't yet in the log file in your email.
                #    They'll only be present in the on-disk logfile.
                try:
                    # This can hang, apparently.
                    # Hangs like this might be fairly damaging, unfortunately.
                    # According to Ken:
                    #   >If sync is trying to write a file down to disk that was deleted,
                    #   >it can hang like that. Unfortunately, the node will have to be
                    #   >power cycled to deal with this situation.
                    #
                    # Let's hope that's not common.
                    # We'll just timeout the ordinary way and hope for the best.
                    subprocess_run("sync", timeout=10.0)
                    time.sleep(2.0)
                except TimeoutExpired:
                    logger.warning("Timed out while waiting for filesystem sync")

                body += "\nLOG (possibly truncated):\n\n"
                with open(f"{logpath}", "r") as log:
                    body += log.read()

            msg = MIMEText(body)
            msg["Subject"] = f"Workflow exited: {result}"
            msg["From"] = f"flyemflows <{user}@{host}>"
            msg["To"] = ",".join(addresses)

            try:
                s = smtplib.SMTP("mail.hhmi.org")
                s.sendmail(msg["From"], addresses, msg.as_string())
                s.quit()
            except:
                msg = (
                    "Failed to send completion email.  Perhaps your machine "
                    "is not configured to send login-less email, which is required for this feature."
                )
                logger.error(msg)

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
    # Duplicate the original fd for restoration
    orig_fd = os.dup(fd)
    # Create a pipe and redirect `fd` into its write-end
    r, w = os.pipe()
    os.dup2(w, fd)
    os.close(w)

    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        result = e
    finally:
        # Restore the original fd
        os.dup2(orig_fd, fd)
        os.close(orig_fd)

    # Read whatever was written into the pipe
    captured = os.read(r, 10_000_000).decode(errors="ignore")
    os.close(r)
    if "error" in captured:
        raise Exception(f"Draco error: {captured}")
    return result, captured
