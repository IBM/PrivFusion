from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage  # type: ignore
from tenacity import retry, stop_after_attempt, wait_fixed  # type: ignore

from privfusion.agents.llms import LLM
from privfusion.data_models import DatasetInformation, SemanticInformation, SemanticRelationship, StructuralInformation
from privfusion.utils import get_type_mapping


class DatasetAnalyzer:
    DESCRIPTION_SYSTEM_PROMPT = """I have a tabular dataset contisting of various columns. Given the column names and some examples as per (SAMPLE) describe the overall semantic of the dataset in a formal style. Only return the discorsive description."""
    COLUMN_SEMANTIC_SYSTEM_PROMPT = """Given the description of a dataset (DESCRIPTION), the column type (COLUMN_TYPE) describe the semantic of the column (COLUMN) in a formal style. Only return the discorsive description."""
    TOPICS_SYSTEM_PROMPT = """Given the dataset descripion (DESCRIPTION) and the sample (SAMPLE) list topics of the dataset. Respond with a list of topics one per line only"""
    RELATIONSHIPS_SYSTEM_PROMPT = """Given the dataset in (SAMPLE) highlight relationships between columns in the following format: {column_name} -> {column_name} := {explanation}. Do not provide any comment."""
    STRUCTURE_DESCRIPTION_SYSTEM_PROMPT = """Given a tabular datasets consisting of columns (COLUMN_NAMES) and some examples as per (SAMPLE), please describe the structure of the dataset to the best of your understanding. Only return the discorsive description."""

    def __init__(self, llm_reference: LLM, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = dict()
        self._llm = llm_reference
        self._description_system_prompt = config.get(
            "description_system_prompt",
            DatasetAnalyzer.DESCRIPTION_SYSTEM_PROMPT,
        )
        self._column_semantic_system_prompt = config.get(
            "column_semantic_system_prompt",
            DatasetAnalyzer.COLUMN_SEMANTIC_SYSTEM_PROMPT,
        )
        self._topics_system_prompt = config.get(
            "topics_system_prompt",
            DatasetAnalyzer.TOPICS_SYSTEM_PROMPT,
        )
        self._relationships_system_prompt = config.get(
            "relationships_system_prompt",
            DatasetAnalyzer.RELATIONSHIPS_SYSTEM_PROMPT,
        )
        self._struct_description_system_prompt = config.get(
            "structure_description_system_prompt",
            DatasetAnalyzer.STRUCTURE_DESCRIPTION_SYSTEM_PROMPT,
        )

    def extract_information_from_dataset(
        self,
        name: str,
        dataset: pd.DataFrame,
    ) -> DatasetInformation:
        structure = self._extract_structure(dataset)
        semantic = self._extract_semantic(dataset)
        relationships = self._extract_relationships(dataset, structure, semantic)

        return DatasetInformation(
            name,
            semantic,
            structure,
            relationships,
        )

    def _extract_semantic(self, dataset: pd.DataFrame) -> SemanticInformation:
        column_types = get_type_mapping(dataset)

        description = self._generate_description(dataset)

        column_semantic = self._generate_column_semantic(
            dataset,
            description,
            column_types,
        )
        dataset_topics = self._extract_topics(dataset, description)

        return SemanticInformation(
            description=description,
            column_types=column_types,
            column_semantic=column_semantic,
            dataset_topics=dataset_topics,
        )

    def _generate_description(self, dataset: pd.DataFrame, n: int = 10) -> str:
        sample_size = min(n, len(dataset))
        response = self._llm.chat(
            [
                SystemMessage(self._description_system_prompt),
                HumanMessage(
                    "This is sample of the dataset in markdown format:\n\n"
                    + str(dataset.sample(n=sample_size).to_markdown()),
                ),
            ],
        )
        return response["response"]

    def _generate_column_semantic(
        self,
        dataset: pd.DataFrame,
        description: str,
        column_types: list[str | None],
    ) -> list[str | None]:
        column_semantic: list[str | None] = []

        for column_name, column_type in zip(dataset.columns, column_types, strict=False):
            chats: list[BaseMessage] = [
                SystemMessage(self._column_semantic_system_prompt),
                HumanMessage("DESCRIPTION: " + description),
                HumanMessage("COLUMN:" + column_name),
            ]
            if column_type is not None:
                chats.append(HumanMessage("COLUMN_TYPE: " + column_type))
            else:
                chats.append(HumanMessage("COLUMN_TYPE: Unknown"))

            column_semantic.append(self._llm.chat(chats)["response"])

        return column_semantic

    def _extract_topics(
        self,
        dataset: pd.DataFrame,
        description: str,
        n: int = 10,
    ) -> list[str]:
        sample_size = min(n, len(dataset))
        response = self._llm.chat(
            [
                SystemMessage(self._topics_system_prompt),
                HumanMessage("SAMPLE:\n" + str(dataset.sample(n=sample_size).to_markdown())),
            ],
        )

        return self._process_topic_response(response["response"])

    def _process_topic_response(self, response: str) -> list[str]:
        return [topic.strip() for topic in response.strip().split("\n")]

    def _extract_structure(self, dataset: pd.DataFrame) -> StructuralInformation:
        return StructuralInformation(
            names=[column_name for column_name in dataset.columns],
            types=[str(dtype) for dtype in dataset.dtypes],
            constraints=[],
            description=self._extract_structure_description(dataset),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _extract_structure_description(self, dataset: pd.DataFrame) -> str:
        sample_size = min(10, len(dataset))
        structure_descripton = self._llm.chat(
            [
                SystemMessage(self._struct_description_system_prompt),
                HumanMessage("COLUMN_NAMES:\n\n" + ",".join(dataset.columns)),
                HumanMessage("SAMPLE:\n\n" + str(dataset.sample(n=sample_size).to_markdown())),
            ],
        )["response"]

        return structure_descripton

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _extract_relationships(
        self,
        dataset: pd.DataFrame,
        structure: StructuralInformation,
        semantic: SemanticInformation,
    ) -> list[SemanticRelationship]:
        sample_size = min(10, len(dataset))
        textual_description = self._llm.chat(
            [
                SystemMessage(self._relationships_system_prompt),
                HumanMessage("SAMPLE:\n\n" + str(dataset.sample(n=sample_size).to_markdown())),
            ],
        )["response"]

        relationships: list[SemanticRelationship] = []

        for line in textual_description.strip().split("\n"):
            # assuming line is of the form: c1 -> c2 := description
            line = line.strip()

            if not len(line):
                continue
            if " := " not in line:
                continue
            if " -> " not in line:
                continue

            c1_c2, reason = line.strip().split(" := ", 1)
            column_from, column_to = c1_c2.split(" -> ")

            relationships.append(SemanticRelationship(column_from, column_to, reason))

        return relationships
