"""Graph-based Retrieval-Augmented Generation (GraphRAG) system."""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Annotated, List, Literal

from graphrag.prompts import PROMPTS
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import GraphCommand, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import Send
from pydantic import BaseModel, Field
from trustcall import create_extractor
import langsmith as ls

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """An entity in the knowledge graph.

    Represents a node in the knowledge graph with a name, type, and description.
    """

    name: str = Field(description="The unique identifier/name of the entity")
    type: str = Field(
        description="The type/category of the entity (e.g., 'person', 'location', 'concept')"
    )
    description: str = Field(description="A detailed description of the entity")


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph.

    Represents an edge in the knowledge graph connecting two entities.
    """

    source: str = Field(description="The name/identifier of the source entity")
    target: str = Field(description="The name/identifier of the target entity")
    type: str = Field(description="The type of relationship between the entities")
    description: str = Field(description="A detailed description of the relationship")


class Graph(BaseModel):
    """A collection of entities and relationships.

    Represents the complete knowledge graph structure containing nodes (entities)
    and edges (relationships) between them.
    """

    entities: List[Entity] = Field(
        description="List of entities (nodes) in the graph", default_factory=list
    )
    relationships: List[Relationship] = Field(
        description="List of relationships (edges) between entities",
        default_factory=list,
    )


class ExtractionStatus(BaseModel):
    """Whether or not to continue gleaning."""

    reason: str = Field(description="Why we are done or still need another pass.")
    status: Literal["done", "continue"] = Field(
        description="done if you are certain all distinct entities and relationship have been extracted, continue otherwise"
    )


class ExtractGraph(Graph):
    """Extract entities and relationships from content."""

    status: ExtractionStatus


@dataclass
class GraphState:
    """The state of the graph processing system.

    Maintains the current state of graph processing including content,
    operation tracking, and conversation history.
    """

    messages: Annotated[list, add_messages]


llm = init_chat_model()

ENTITY_TYPES = [
    "person",
    "location",
    "organization",
    "concept",
    "event",
    "product",
    "work_of_art",
    "technology",
    "skill",
    "industry",
    "time_period",
    "natural_phenomenon",
    "biological_entity",
    "legal_entity",
    "financial_instrument",
    "other",
]


@dataclass
class ExtractionState:
    """Send payload."""

    chunk_id: int
    chunk: str

@ls.traceable
async def _load_in_progress(
    user_id: str, thread_id: str, chunk_id: str, store: BaseStore
) -> Graph:
    """Load current extraction results related to this thread."""
    # If a thread is re-activated after some time, we may have already
    # performed extraction for it. In that case, we want to load the existing
    # results.
    coros = []

    relationship_items = await store.asearch(
        ("graph", user_id, "relationships", "*", thread_id),
        filter={"chunk_id": {"$eq": chunk_id}},
    )
    _relationship_items = await store.asearch(
        ("graph", user_id, "relationships", "*", thread_id),
        # filter={"chunk_id": {"$eq": chunk_id}},
    )
    
    if not relationship_items:
        return None
    # Collect all matching entities
    unique_names = set()
    for rel in relationship_items:
        unique_names.add(rel.value["source"])
        unique_names.add(rel.value["target"])

    for name in unique_names:
        coros.append(
            store.asearch(
                ("graph", user_id, "entities"),
                filter={"name": {"$eq": name}},
            )
        )

    entity_items = await asyncio.gather(*coros)
    relationships = []
    for it in relationship_items:
        relationships.append(
            Relationship(
                source=rel.value["source"],
                target=rel.value["target"],
                type=rel.value["type"],
                description=rel.value["description"],
                chunk_id=rel.value["chunk_id"],
            )
        )

    entities = []
    for items in entity_items:
        for item in items:
            entities.append(
                Entity(
                    name=item.value["name"],
                    type=item.value["type"],
                    description=item.value["description"],
                )
            )

    return Graph(entities=entities, relationships=relationships)


async def extract_chunk(
    state: ExtractionState, config: RunnableConfig, *, store: BaseStore
):
    """Extract entities and relationships from content."""
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    messages = [
        {
            "role": "system",
            "content": PROMPTS["entity_relationship_extraction"]["system"],
        },
        {
            "role": "human",
            "content": PROMPTS["entity_relationship_extraction"]["human"].format(
                content=state.chunk
            ),
        },
    ]
    in_progress = await _load_in_progress(user_id, thread_id, state.chunk_id, store)
    if in_progress:
        current_graph = in_progress
    else:
        extractor = create_extractor(llm, tools=[Graph], tool_choice="any")
        initial_result = await extractor.ainvoke({"messages": messages})
        current_graph: Graph = initial_result["responses"][0]
    gleaner = create_extractor(llm, tools=[ExtractGraph], tool_choice="any")

    # Get valid entity types

    clean_entity_types = [re.sub(r"[ _]", "", et).upper() for et in ENTITY_TYPES]

    # Track valid entity names for relationship validation
    valid_entities = {entity.name for entity in current_graph.entities}
    for _ in range(3):
        gleaning_result = await gleaner.ainvoke(
            {
                "messages": messages,
                "existing": {"ExtractGraph": current_graph.model_dump(mode="json")},
            }
        )

        current_graph: ExtractGraph = gleaning_result["responses"][0]
        for entity in current_graph.entities:
            if (
                not entity.type
                or re.sub(r"[ _]", "", entity.type).upper() not in clean_entity_types
            ):
                entity.type = "UNKNOWN"
            valid_entities.add(entity.name)

        if current_graph.status.status == "done":
            break
    coros = []
    for entity in current_graph.entities:
        coros.append(
            store.aput(
                ("graph", user_id, "entities", entity.type.lower()),
                entity.name,
                entity.model_dump(mode="json"),
            )
        )

    for relationship in current_graph.relationships:
        rel_key = (
            f"{relationship.source}:{relationship.target}:{relationship.type}".strip()
            .lower()
            .replace(" ", "_")
        )
        coros.append(
            store.aput(
                (
                    "graph",
                    user_id,
                    "relationships",
                    relationship.type.lower(),
                    # Dedup relationships by thread_id
                    thread_id,
                ),
                rel_key,
                {
                    **relationship.model_dump(mode="json"),
                    "chunk_id": state.chunk_id,
                },
                index=False,
            )
        )

    coros.append(
        store.aput(
            ("graph", user_id, "chunks", str(thread_id)),
            str(state.chunk_id),
            {"text": state.chunk},
        )
    )

    await asyncio.gather(*coros)


async def handle_information_extraction(
    state: GraphState, config: RunnableConfig, *, store: BaseStore
) -> dict:
    """Extract entities and relationships from content."""
    user_id = config["configurable"]["user_id"]
    contents = []
    for message in state.messages:
        if message.type == "human":
            name = f"name={message.name}" if message.name else ""
            contents.append(f"""<user id="{user_id}{name}">{message.content}</user>""")
        elif message.type == "ai":
            contents.append(f"""<assistant>{message.content}</assistant>""")
        else:
            continue

    content = "\n\n".join(contents)
    chunk_size = 2000
    chunk_overlap = 0.25
    chunks = [
        content[i : i + chunk_size]
        for i in range(0, len(content), int(chunk_size * (1 - chunk_overlap)))
    ]
    sends = [
        Send("extract_chunk", ExtractionState(chunk_id=i, chunk=chunk))
        for i, chunk in enumerate(chunks)
    ]
    return GraphCommand(send=sends)


builder = StateGraph(GraphState)

builder.add_node(handle_information_extraction)
builder.add_node(extract_chunk)
builder.add_edge("__start__", "handle_information_extraction")
graph = builder.compile()
