#!/usr/bin/env python
import os
import sys

from io import StringIO

import pint.scripts.zima as zima
from pinttestdata import datadir


def test_result(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = tmp_path / "fake_testzima.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"{parfile} {timfile}"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout


def test_wb_result(tmp_path):
    parfile = os.path.join(datadir, "NGC6440E.par")
    timfile = tmp_path / "fake_testzima_wb.tim"
    saved_stdout, sys.stdout = sys.stdout, StringIO("_")
    try:
        cmd = f"{parfile} {timfile} --addnoise --simulatedm --dmerror 1e-5"
        zima.main(cmd.split())
        lines = sys.stdout.getvalue()
    finally:
        sys.stdout = saved_stdout
