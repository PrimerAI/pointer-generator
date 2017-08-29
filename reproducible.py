""" Assist in making experiments reproducible.

    Usage:

      # The import sets the random seed, and records both the seed and the current git commit.

      import reproducible

      reproducible.add_non_git_file('my_data.json')  # Record a hash of this file.
      reproducible.write_fingerprint()               # Print a reproducibility fingerprint.

    The call to write_fingerprint() will exit the process with a message if the current git tree is
    dirty (that is, if any tracked files are edited vs the current commit, or if any local Python
    files are present but not tracked). Those checks can be skipped by setting this boolean:

      reproducible.allow_git_edits = False

    The goal of a "reproducibility fingerprint" is to make it easy to recreate the conditions of
    the current run. This includes the current git commit (which is enforced by checking that local
    tracked files have not been changed), the random seed, and hashes of any used data files.
"""

# TODO(tyler): Add future-time checks for the hash values based on data files.

import hashlib
import os
import random
import subprocess
import sys
import textwrap


FILE_BUFFER_SIZE = 65536


commit = None
file_hashes = {}
allow_git_edits = False


def add_non_git_file(filename):
    """Register a file as being used so future users can verify that they have the same data."""

    global file_hashes

    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(FILE_BUFFER_SIZE)
            if not data:
                break
            sha1.update(data)

    file_hashes[filename] = sha1.hexdigest()


def write_fingerprint(f=sys.stdout):
    """Write out a reproducibility fingerprint to stdout (default) or a given file-like object."""

    global commit, file_hashes, seed

    git_err = _check_if_git_is_clean()

    fingerprint_fmt = """\
            %s
            Reproducibility fingerprint:

            # Use these shell commands to reproduce this run:%s
            git checkout %s
            SEED=%d python %s
            %s
            %s
            """
    fingerprint_fmt = textwrap.dedent(fingerprint_fmt)

    if file_hashes:
        hash_list = ['  %s: %s' % (name, sha1) for name, sha1 in file_hashes.items()]
        file_msg = '\nFile hashes:\n' + '\n'.join(hash_list)
    else:
        file_msg = ''

    separator = '------------------------------------------------------------------'

    git_msg = ''
    if git_err:
        git_msg = '\n# (Warning from original run: %s)' % git_err

    cmd = ' '.join(sys.argv)


    fingerprint = fingerprint_fmt % (separator, git_msg, commit, seed, cmd, file_msg, separator)

    f.write(fingerprint)


def _freak_out_and_exit(msg):
    """Exit and print out an informative error message explaining why we chose to do so."""

    sys.stderr.write('Error from reproducible.py: %s\n' % msg)
    sys.exit(1)


def _init():
    """Perform import-time initialization."""

    global commit, seed

    # Avoid running more than once.
    if commit:
        return

    # Set and record the random seed.
    seed = int(os.getenv('SEED') or hash(os.urandom(10)))
    random.seed(seed)

    # Extract the current git commit.
    git_output = subprocess.check_output(['git', 'show', '--oneline'])
    commit = git_output.split()[0]  # The first token is a short commit hash.


def _check_if_git_is_clean():
    """Check to see if the current git repo is clean.

    If allow_git_edits is False, exit the process when the repo is dirty.
    Otherwise, return an error message string when dirty, or None when clean.
    """

    global allow_git_edits

    err_msg = None

    # Check for changes to tracked files.
    return_code = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'])
    if return_code != 0:
        err_msg = 'A git-tracked file has been edited.'

    # Check for untracked Python files.
    git_status_lines = subprocess.check_output(['git', 'status', '--porcelain']).split('\n')
    python_lines = [line for line in git_status_lines if line.endswith('.py')]
    if not err_msg and python_lines:
        err_msg = 'The Python file %s is untracked by git.' % python_lines[0][3:]

    if err_msg and not allow_git_edits:
        _freak_out_and_exit(err_msg)

    return err_msg

_init()
