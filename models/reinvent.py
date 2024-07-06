#!/usr/bin/env python
"""Main entry point into Reinvent."""

from __future__ import annotations

import datetime
import getpass
import os
import platform
import random
import subprocess as sp
import sys
import uuid
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from .base import BaseGenerator

SYSTEM = platform.system()

if SYSTEM != "Windows":
    import resource  # Unix only

import numpy as np
import rdkit
import torch
from rdkit import RDLogger, rdBase
from reinvent import config_parse, runmodes, setup_logger, version
from reinvent.runmodes.reporter.remote import setup_reporter
from reinvent.runmodes.utils import set_torch_device

rdBase.DisableLog("rdApp.*")


def enable_rdkit_log(levels: List[str]):
    """Enable logging messages from RDKit for a specific logging level.

    :param levels: the specific level(s) that need to be silenced
    """

    if "all" in levels:
        RDLogger.EnableLog("rdApp.*")
        return

    for level in levels:
        RDLogger.EnableLog(f"rdApp.{level}")


def get_cuda_driver_version() -> Optional[str]:
    """Get the CUDA driver version via modinfo if possible.

    This is for Linux only.

    :returns: driver version or None
    """

    try:
        result = sp.run(["/sbin/modinfo", "nvidia"], shell=False, capture_output=True)
    except Exception:
        return

    for line in result.stdout.splitlines():
        str_line = line.decode()

        if str_line.startswith("version:"):
            cuda_driver_version = str_line.split()[1]
            return cuda_driver_version


def set_seed(seed: int):
    """Set global seed for reproducibility

    :param seed: the seed to initialize the random generators
    """

    if seed is None:
        return

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Reinvent(BaseGenerator):
    INPUT_FORMAT = "toml"
    DEVICE = "cpu"
    LOG_FILENAME = None
    LOG_LEVEL = "info"
    SEED = None
    DOTENV_FILENAME = None
    ENABLE_RDKIT_LOG_LEVELS = None
    RESPONDER_TOKEN = "RESPONDER_TOKEN"

    def __init__(self, config_filename: str):
        self.config_filename = os.path.abspath(config_filename)
        self.config = None

    def setup_responder(self, config):
        """Setup for remote monitor

        :param config: configuration
        """

        endpoint = config.get("endpoint", False)

        if not endpoint:
            return

        token = os.environ.get(self.RESPONDER_TOKEN, None)
        setup_reporter(endpoint, token)

    def generate(self, smiles: List[str]) -> List[str]:
        output_file = self._call_reinvent(smiles)
        return self._read_output_file(output_file)

    def _read_output_file(self, output_file: str) -> List[str]:
        df = pd.read_csv(output_file)
        return df["SMILES"].tolist()

    def _create_input_file(self, smiles: List[str]) -> str:
        run_uuid = str(uuid.uuid4().hex)
        input_file = self.config_filename.replace(".toml", f"{run_uuid}_input.smi")
        with open(input_file, "w") as f:
            f.write("\n".join(smiles))
        return input_file

    def _call_reinvent(self, smiles: List[str]) -> str:
        dotenv_loaded = load_dotenv(self.DOTENV_FILENAME)

        reader = getattr(config_parse, f"read_{self.INPUT_FORMAT}")
        self.config = reader(self.config_filename)

        # Overwrite the smiles_file and output_file in the config file
        assert self.config.get("smiles_file") is None, "smiles_file is not allowed in the config file"
        assert self.config.get("output_file") is None, "output_smiles is not allowed in the config file"
        input_file = self._create_input_file(smiles)
        self.config["parameters"]["smiles_file"] = input_file
        self.config["parameters"]["output_file"] = input_file.replace(".smi", "_output.smi")

        if self.ENABLE_RDKIT_LOG_LEVELS:
            enable_rdkit_log(self.ENABLE_RDKIT_LOG_LEVELS)

        run_type = self.config["run_type"]
        runner = getattr(runmodes, f"run_{run_type}")
        logger = setup_logger(name=__package__, level=self.LOG_LEVEL.upper(), filename=self.LOG_FILENAME)

        have_version = self.config.get("version", version.__config_version__)

        if have_version < version.__config_version__:
            msg = f"Need at least version 4. Input file is for version {have_version}."
            logger.fatal(msg)
            raise RuntimeError(msg)

        logger.info(
            f"Started {version.__progname__} {version.__version__} {version.__copyright__} on "
            f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
        )

        logger.info(f"Command line: {' '.join(sys.argv)}")

        if dotenv_loaded:
            logger.info("Environment loaded from dotenv file")

        logger.info(f"User {getpass.getuser()} on host {platform.node()}")
        logger.info(f"Python version {platform.python_version()}")
        logger.info(f"PyTorch version {torch.__version__}, git {torch.version.git_version}")
        logger.info(f"PyTorch compiled with CUDA version {torch.version.cuda}")
        logger.info(f"RDKit version {rdkit.__version__}")
        logger.info(f"Platform {platform.platform()}")

        if cuda_driver_version := get_cuda_driver_version():
            logger.info(f"CUDA driver version {cuda_driver_version}")

        logger.info(f"Number of PyTorch CUDA devices {torch.cuda.device_count()}")

        if "use_cuda" in self.config:
            logger.warning("'use_cuda' is deprecated, use 'device' instead")

        device = self.config.get("device", None)

        if not device:
            use_cuda = self.config.get("use_cuda", True)

            if use_cuda:
                device = "cuda:0"

        actual_device = set_torch_device(self.DEVICE, device)

        if actual_device.type == "cuda":
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"Using CUDA device:{current_device} {device_name}")

            free_memory, total_memory = torch.cuda.mem_get_info()
            logger.info(f"GPU memory: {free_memory // 1024**2} MiB free, " f"{total_memory // 1024**2} MiB total")
        else:
            logger.info(f"Using CPU {platform.processor()}")

        seed = self.config.get("seed", None)

        if self.SEED is not None:
            set_seed(seed)
            logger.info(f"Set seed for all random generators to {seed}")

        tb_logdir = None

        if "tb_logdir" in self.config:
            tb_logdir = os.path.abspath(self.config["tb_logdir"])
            logger.info(f"Writing TensorBoard summary to {tb_logdir}")

        if "json_out_config" in self.config:
            json_out_config = os.path.abspath(self.config["json_out_config"])
            logger.info(f"Writing JSON config file to {json_out_config}")
            config_parse.write_json(self.config, json_out_config)

        responder_config = None

        if "responder" in self.config:
            self.setup_responder(self.config["responder"])
            responder_config = self.config["responder"]
            logger.info(
                f"Using remote monitor endpoint {self.config['responder']['endpoint']} "
                f"with frequency {self.config['responder']['frequency']}"
            )

        runner(self.config, actual_device, tb_logdir, responder_config)

        if SYSTEM != "Windows":
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mem = 0

            if SYSTEM == "Linux":
                peak_mem = maxrss / 1024
            elif SYSTEM == "Darwin":  # MacOSX
                peak_mem = maxrss / 1024**2

            if peak_mem:
                logger.info(f"Peak main memory usage: {peak_mem:.3f} MiB")

        logger.info(f"Finished {version.__progname__} on {datetime.datetime.now().strftime('%Y-%m-%d')}")

        return self.config["parameters"]["output_file"]
