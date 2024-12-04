"""Tests for the GraphRAG system."""

import uuid

import langsmith as ls
import pytest
from chatbot.graph import query_knowledge_graph
from graphrag.graph import builder
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore


@pytest.mark.asyncio
@ls.unit
async def test_basic_entity_extraction():
    """Test basic entity extraction from text."""
    store = InMemoryStore(
        index={"embed": OpenAIEmbeddings(model="text-embedding-3-small"), "dims": 1536}
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"model": "openai:gpt-4o-mini", "thread_id": thread_id}}

    content = "Alice is a software engineer who loves coding in Python."
    graph = builder.compile(store=store)

    await graph.ainvoke({"messages": [{"role": "user", "content": content}]}, config)

    items = await store.asearch(("graph", "entities"))

    # Should find at least Alice (PERSON), Python (TECHNOLOGY), and Software Engineering (SKILL)
    assert len(items) >= 3

    # Verify Alice is stored as a person
    alice_items = await store.asearch(
        ("graph", "entities"),
        filter={"name": {"$eq": "Alice"}},
    )
    assert len(alice_items) == 1
    assert alice_items[0].value["type"].lower() == "person"

    all_rels = await store.asearch(("graph", "relationships"))

    assert len(all_rels) >= 2

    chunk = await store.aget(("graph", "chunks", thread_id), "0")
    assert chunk is not None
    assert "Alice" in chunk.value


@pytest.mark.asyncio
@ls.unit
async def test_relationship_extraction():
    """Test relationship extraction and gleaning."""
    store = InMemoryStore(
        index={"embed": OpenAIEmbeddings(model="text-embedding-3-small"), "dims": 1536}
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"model": "openai:gpt-4o-mini", "thread_id": thread_id}}

    # Test content
    content = """Bob is a data scientist who works extensively with TensorFlow.
    He's also learning PyTorch for a new ML project with his team."""

    graph = builder.compile(store=store)
    await graph.ainvoke({"messages": [{"role": "user", "content": content}]}, config)

    # Verify entities were stored with correct types
    bob_items = await store.asearch(
        ("graph", "entities"),
        filter={"name": {"$eq": "Bob"}},
    )
    assert len(bob_items) == 1
    assert bob_items[0].value["type"].lower() == "person"

    tech_items = await store.asearch(("graph", "entities"))
    names = {item.value["name"] for item in tech_items}
    assert "TensorFlow" in names
    assert "PyTorch" in names

    # Verify relationships
    work_rels = await store.asearch(("graph", "relationships"))
    learn_rels = [r for r in work_rels if "learning" in r.value["type"].lower()]
    work_rels = [r for r in work_rels if "works_with" in r.value["type"].lower()]

    assert len(work_rels) >= 1
    assert len(learn_rels) >= 1

    for rel in work_rels + learn_rels:
        assert len(rel.value["chunks"]) > 0
        assert len(rel.value["chunks"][0]) == 2  # [thread_id, chunk_id]


@pytest.mark.asyncio
@ls.unit
async def test_error_handling():
    """Test error handling in graph processing."""
    store = InMemoryStore(
        index={"embed": OpenAIEmbeddings(model="text-embedding-3-small"), "dims": 1536}
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"model": "openai:gpt-4o-mini", "thread_id": thread_id}}

    content = """Alice is a senior software engineer who mentors Bob.
    Bob is learning Python and TypeScript for web development.
    Charlie collaborates with Alice on cloud architecture."""

    graph = builder.compile(store=store)
    await graph.ainvoke({"messages": [{"role": "user", "content": content}]}, config)

    # Query focusing on mentorship
    result = await query_knowledge_graph(
        query="Who is mentoring junior developers?",
        store=store,
        limit=5,
        semantic_weight=0.7,
    )

    assert len(result["entities"]) > 0
    assert len(result["relationships"]) > 0
    assert len(result["sources"]) > 0

    # Verify Alice appears in results with high score
    alice_found = False
    for entity in result["entities"]:
        if entity.value["name"] == "Alice":
            alice_found = True
            break
    assert alice_found

    # Query focusing on technologies
    tech_result = await query_knowledge_graph(
        query="What programming languages are being used?",
        store=store,
        limit=5,
    )

    tech_names = {entity.value["name"] for entity in tech_result["entities"]}
    assert "Python" in tech_names
    assert "TypeScript" in tech_names
