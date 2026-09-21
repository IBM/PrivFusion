"""
Some utility methods for Tabular Data Generation
"""

import json
import logging
from typing import Any

import litellm
import pandas as pd
import yaml
from jinja2 import StrictUndefined, Template

# Define templates
GENERATE_TEMPLATE = """{introduction}\n{principles}\n{examples}\n{generation}"""


def read_data(
    file_path: str,
    sep: str = ",",
    compression: str = "infer",
) -> pd.DataFrame:
    logging.info(f"Data will be loaded from {file_path}")
    df = pd.read_csv(file_path, sep=sep, compression=compression)
    return df


def extract_json_as_dict(json_file: str | dict[str, Any] | list[Any]) -> Any:
    if isinstance(json_file, dict | list):
        return json_file  # If already a dictionary or list, return as-is
    try:
        return json.loads(json_file)  # Try parsing if it's a string
    except (ValueError, json.JSONDecodeError):
        print("JSON decode error")
        print(json_file)
        return None


def prompt_model(
    model: str,
    prompt: str,
    role: str,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_headers: dict[str, Any] | None = None,
) -> str:
    response = litellm.completion(
        model=model,
        messages=[
            {"role": f"{role}", "content": f"{prompt}\n"},
        ],
        api_base=api_base,
        api_key=api_key,
        extra_headers=extra_headers,
    )

    return response.choices[0].message["content"]


def from_yaml(yaml_path: str) -> str:
    """
    Read the prompt template

    :param yaml_path: Path of yaml file

    :return: Template prompt
    """
    with open(yaml_path, encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
    template = GENERATE_TEMPLATE.format(**yaml_config)
    return template


def encode_prompt(prompt: str, render_dict: dict[str, Any]) -> str:
    """
    Encode the prompt template

    :param prompt: Template prompt
    :param render_dict: Dictionary of the placeholder's values

    :return: Template prompt with corresponding values in the placeholders
    """
    return Template(prompt).render(render_dict, undefined=StrictUndefined)
