import argparse
import json
import os
from typing import TypeVar, Type

import smart_open

from deepcell_imaging.gcp_batch_jobs import get_batch_indexed_task, ComputeConfig
from deepcell_imaging.gcp_batch_jobs.types import EnvironmentConfig

ArgType = TypeVar("ArgType")


def get_task_arguments(
    task_name, args_cls: Type[ArgType]
) -> tuple[ArgType, EnvironmentConfig]:
    parser = argparse.ArgumentParser(task_name, add_help=False)

    parser.add_argument(
        "--tasks_spec_uri",
        help="URI to a JSON file containing a list of task parameters. Specify this, OR, the individual parameters.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--env_config_uri",
        help="URI to a JSON file containing environment configuration",
        type=str,
        required=False,
    )

    env_config = None

    # If needed, we could change this to act on an args list passed
    # as a parameter, instead of just using implicit sys.argv
    parsed_args, args_remainder = parser.parse_known_args()

    if parsed_args.env_config_uri:
        with smart_open.open(parsed_args.env_config_uri, "r") as env_config_file:
            env_config = EnvironmentConfig(**json.load(env_config_file))

    if parsed_args.tasks_spec_uri:
        if len(args_remainder) > 0:
            raise ValueError("Either pass --tasks_spec_uri alone, or not at all")

        return get_batch_indexed_task(parsed_args.tasks_spec_uri, args_cls), env_config
    else:
        model_fields = args_cls.model_fields

        parser = argparse.ArgumentParser(task_name, parents=[parser])

        for key in model_fields.keys():
            field = model_fields[key]
            parser.add_argument(
                f"--{key}",
                help=field.description,
                type=field.annotation,
                required=field.is_required(),
            )

        parsed_args = parser.parse_args(args_remainder)
        kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
        return args_cls(**kwargs), env_config


def add_dataset_parameters(
    parser: argparse.ArgumentParser, require_measurement_parameters: bool
) -> None:
    subparsers = parser.add_subparsers(help="Mode of operation", dest="mode")

    workspace_parser = subparsers.add_parser("workspace")
    workspace_parser.add_argument(
        "dataset_path",  # positional argument
        help="Path to the dataset",
    )
    workspace_parser.add_argument(
        "--images_subdir",
        help="Subdirectory within the dataset containing the images",
        type=str,
        default="OMETIFF",
    )
    workspace_parser.add_argument(
        "--npz_subdir",
        help="Subdirectory within the dataset containing the image files as numpy arrays",
        type=str,
        default="NPZ_INTERMEDIATE",
    )
    workspace_parser.add_argument(
        "--segmasks_subdir",
        help="Subdirectory within the dataset containing the segmentation masks",
        type=str,
        default="SEGMASK",
    )
    workspace_parser.add_argument(
        "--project_subdir",
        help="Subdirectory within the dataset containing the QuPath project",
        type=str,
        default="PROJ",
    )
    workspace_parser.add_argument(
        "--reports_subdir",
        help="Subdirectory within the dataset containing the QuPath reports",
        type=str,
        default="REPORTS",
    )

    paths_parser = subparsers.add_parser("paths")
    paths_parser.add_argument(
        "--images_path",
        help="Path to the images",
        required=True,
    )
    paths_parser.add_argument(
        "--numpy_path",
        help="Path to the images as numpy arrays",
        required=True,
    )
    paths_parser.add_argument(
        "--segmasks_path",
        help="Path to the segmentation masks",
        required=True,
    )
    paths_parser.add_argument(
        "--project_path",
        help="Path to the QuPath project",
        required=True if require_measurement_parameters else False,
    )
    paths_parser.add_argument(
        "--reports_path",
        help="Path to the QuPath reports",
        required=True if require_measurement_parameters else False,
    )


def get_dataset_paths(args) -> dict:
    if args.mode == "workspace":
        image_root = os.path.join(args.dataset_path, args.images_subdir)
        npz_root = os.path.join(args.dataset_path, args.npz_subdir)
        masks_output_root = os.path.join(args.dataset_path, args.segmasks_subdir)
        project_root = os.path.join(args.dataset_path, args.project_subdir)
        reports_root = os.path.join(args.dataset_path, args.reports_subdir)
    elif args.mode == "paths":
        image_root = args.images_path
        npz_root = args.numpy_path
        masks_output_root = args.segmasks_path
        project_root = args.project_path
        reports_root = args.reports_path
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    return {
        "image_root": image_root,
        "npz_root": npz_root,
        "masks_output_root": masks_output_root,
        "project_root": project_root,
        "reports_root": reports_root,
    }


# Parse strings in the format:
# machine_type:provisioning_model+accelerator_type:accelerator_count
# where each part is optional.
#
# Defaults:
#   - machine_type: n1-standard-8
#   - provisioning_model: SPOT
#   - accelerator_type: nvidia-tesla-t4
#   - accelerator_count: 1
def parse_compute_config(compute_str: str) -> ComputeConfig:
    compute_parts = compute_str.split("+")

    if len(compute_parts) == 1:
        compute_parts = [compute_parts[0], ""]

    machine_parts = compute_parts[0].split(":")
    if len(machine_parts) == 1:
        machine_type = machine_parts[0]
        provisioning_model = "SPOT"
    elif len(machine_parts) == 2:
        machine_type, provisioning_model = machine_parts
    else:
        raise ValueError(f"Invalid machine type/provisioning model: {machine_parts}")

    accelerator_parts = compute_parts[1].split(":")
    if len(accelerator_parts) == 1:
        accelerator_type = accelerator_parts[0]
        if accelerator_type:
            accelerator_count = 1
        else:
            accelerator_count = 0
    elif len(accelerator_parts) == 2:
        accelerator_type = accelerator_parts[0]
        accelerator_count = int(accelerator_parts[1] or 0)

        if accelerator_count > 0 and not accelerator_type:
            raise ValueError(f"Accelerator count specified without accelerator type")
    else:
        raise ValueError(f"Invalid accelerator type/count: {accelerator_parts}")

    return ComputeConfig(
        machine_type=machine_type or "n1-standard-4",
        provisioning_model=provisioning_model or "SPOT",
        accelerator_type=accelerator_type or "",
        accelerator_count=int(accelerator_count or 0),
    )
