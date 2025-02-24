#!/usr/bin/env python
"""
Script to launch a QuPath measurement job on Batch.

This script can be used to launch a QuPath job on its own, or as the last step
of a segmentation job (effectively implementing dependencies).
"""

import json
import os
import subprocess
import tempfile
import uuid

import deepcell_imaging.gcp_logging
import logging

from deepcell_imaging.gcp_batch_jobs.quantify import (
    make_quantify_job,
)
from deepcell_imaging.gcp_batch_jobs.types import EnqueueQuantifyArgs
from deepcell_imaging.utils.cmdline import get_task_arguments, parse_compute_config


def main():
    deepcell_imaging.gcp_logging.initialize_gcp_logging()
    logger = logging.getLogger(__name__)

    try:
        task_index = int(os.environ["BATCH_TASK_INDEX"])
    except KeyError:
        logger.info("Not running on batch? Assuming task index 0.")
        task_index = 0

    if task_index != 0:
        logger.info(f"Skipping task {task_index}; we only run the first task.")
        return

    args, env_config = get_task_arguments(
        "launch_qupath_measurement", EnqueueQuantifyArgs
    )

    if not env_config:
        raise ValueError("Environment configuration is required")

    container_image = env_config.quantify_container_image
    region = env_config.region

    job_json = make_quantify_job(
        region=region,
        container_image=container_image,
        args=args,
        networking_interface=env_config.networking_interface,
        service_account=env_config.service_account,
    )

    job_json_file = tempfile.NamedTemporaryFile()
    with open(job_json_file.name, "w") as f:
        json.dump(job_json, f)

    # The batch job id must be unique, and can only contain lowercase letters,
    # numbers, and hyphens. It must also be 63 characters or fewer.
    # We're doing 62 to be safe.
    #
    # Regex: ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$
    batch_job_id = "qupath-{}".format(str(uuid.uuid4()))
    batch_job_id = batch_job_id[0:62].lower()

    cmd = "gcloud batch jobs submit {job_id} --location {location} --config {job_filename}".format(
        job_id=batch_job_id, location=region, job_filename=job_json_file.name
    )
    subprocess.run(cmd, shell=True)

    logger.info("Job submitted with ID: {}".format(batch_job_id))


if __name__ == "__main__":
    main()
