import os
from pathlib import Path
from typing import Any

import pandas as pd

from privfusion.tabular_data import utils

VERSION = "0.0.0"


def generate_tabular_data(
    icl_examples: list,
    model: str,
    NUM_PROMPT_INSTRUCTIONS: int,
    role: str = "user",
    templates: str | Path = "templates/adult_generation.yaml",
    **kwargs: Any,
) -> pd.DataFrame:
    if not kwargs:
        kwargs = {}
    synth_data = []
    prompt_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        templates,
    )
    adult_prompt_gen = utils.from_yaml(prompt_file_path)

    for i in range(0, len(icl_examples), NUM_PROMPT_INSTRUCTIONS):
        prompt_examples = icl_examples[i : i + NUM_PROMPT_INSTRUCTIONS]
        examples = "\n".join(
            [f"### User {ex_i}: {example}" for ex_i, example in enumerate(prompt_examples, start=1)],
        )

        adult_gen_inp_dict = {
            "icl_examples": examples,
        }
        input_prompts = utils.encode_prompt(adult_prompt_gen, adult_gen_inp_dict)

        llm_outputs = utils.prompt_model(
            model=model,
            prompt=input_prompts,
            role=role,
            api_base=kwargs.get("api_base"),
            extra_headers=kwargs.get("extra_headers"),
            api_key=kwargs.get("api_key"),
        )
        records = utils.extract_json_as_dict(llm_outputs)
        if records:
            for record in records:
                synth_data.append(pd.DataFrame([record]))
        else:
            print("Skipping empty response…")
    gen_synth_data = pd.concat(synth_data, axis=0, ignore_index=True)
    return gen_synth_data
