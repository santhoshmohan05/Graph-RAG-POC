import asyncio
import nest_asyncio
import re

nest_asyncio.apply()

from typing import Any, List, Callable, Optional, Union, Dict

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field
import json

import logging
logger = logging.getLogger("my_logger")


KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets. 

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Format the Output in the following manner
    - Each line of output is a proper json object
    - Each line is either a Entity json object or a Relationship json object
    - each line is separated by a single new line character
    - do Not any any extra lines apart from the formatted output
    - the source_entity and the target_entity should be from the entity_name that is mentioned in entity objects
   Example Output:
{"entity_name":"Alice","entity_type":"Person","entity_description":"Alice is a person belonging to the family"}
{"entity_name":"Bob","entity_type":"Person","entity_description":"Bob is a person belonging to the family"}
{"entity_name":"Summer Cottage","entity_type":"Place","entity_description":"Summer Cottage is a house in vermont belonging to the family"}
{"source_entity":"Alice","target_entity":"Bob","relation":"Daughter","relationship_description":"Alice is the daughter of Bob"}
{"source_entity":"Bob","target_entity":"Summer Cottage","relation":"Owner","relationship_description":"Bob owns the Summer cottage in vermont"}

-Context-
The text corpus is from a technical document that details Best Practices for SAP HANA on VMware vSphere.

-Rules-
1. DO NOT Invent new relationships. Try to get it from the context of text
2. Try to create the relationship description from the provided text context
3. Think step-by-step about entites and transistive proerties to establish complex relationship such as mutual exclusivity

-Real Data-
######################
text: {text}
######################
output:"""

def parse_fn_working(response_str: str) -> Any:
    logger.info("parser running for job")
    entities = []
    relationships = []
    lines = response_str.strip().split("\n")
    for line in lines:
        line = line.strip()
        try:
            if line.startswith("{") and line.endswith("}"):
                obj:dict = json.loads(line)
                if "entity_name" in obj:
                    entities.append((obj.get("entity_name"),obj.get("entity_type","Object"),obj.get("entity_description","")))
                elif "source_entity" in obj and "target_entity" in obj:
                    relationships.append((obj.get("source_entity"),obj.get("target_entity"), obj.get("relation",""), obj.get("relationship_description","")))
            else:
                logger.error(f"faulty line - {line}")
        except Exception as e:
            logger.error(f"{line} - {str(e)}")
    logger.info(f"passing {len(entities)} entities and {len(relationships)} relationships")
    return entities, relationships

class GraphRAGExtractor(TransformComponent):
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_fn_working,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings
        extract_prompt_new = PromptTemplate(KG_TRIPLET_EXTRACT_TMPL)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt_new,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )
    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )
    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")
        logger.info("Extract running for node")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError as e:
            logger.exception(e)
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))
        print(f"{len(jobs)} nodes are going to run")
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
