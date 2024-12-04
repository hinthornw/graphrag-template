"""Define utility functions for your graph."""

import asyncio
import logging
import math
from collections import defaultdict
from typing import Dict, List, TypedDict

import langsmith as ls
from langgraph.store.base import BaseStore, Item, SearchItem
from pydantic import Field

logger = logging.getLogger(__name__)


class QueryResult(TypedDict):
    """Result of a graph query including scored entities and relationships."""

    entities: List[SearchItem] = Field(description="Scored relevant entities")
    relationships: List[tuple[dict, float]] = Field(
        description="Scored relevant relationships"
    )
    sources: List[Item] = Field(description="Sources of the query")


@ls.traceable
async def query_knowledge_graph(
    query: str,
    user_id: str,
    store: BaseStore,
    limit: int = 10,
    semantic_weight: float = 0.7,
) -> QueryResult:
    """Query the graph using natural language with optional type filtering."""
    try:
        # Step 1: Find relevant entities via vector search
        async with ls.trace("vector_search", inputs={"query": query}) as rt:
            all_results = await store.asearch(
                ("graph", user_id, "entities"), query=query, limit=limit * 3
            )
            rt.add_outputs({"results": all_results})
        if not all_results:
            return {"entities": [], "relationships": [], "sources": []}

        # Step 2: Find relationships connected to top entities
        entity_names = {item.key: item for item in all_results}

        # Get all relationships for PageRank computation
        all_rels: List[List[Item]] = await asyncio.gather(
            *[
                store.asearch(
                    ("graph", user_id, "relationships"),
                    filter=filt,
                    limit=500,  # TODO: Make this configurable
                )
                for name in entity_names
                for filt in [
                    {"source": {"$eq": name}},
                    {"target": {"$eq": name}},
                ]
            ]
        )
        if not all_rels:
            return {"entities": [], "relationships": [], "sources": []}

        flat_rels = [rel for sublist in all_rels for rel in sublist]
        pagerank_scores = compute_pagerank(flat_rels)

        relationship_scores = {}  # (source, target, type) -> list[relationship]
        async with ls.trace("score_relationships", inputs={"relationships": flat_rels}):
            for rel_item in flat_rels:
                rel = rel_item.value
                # Combine semantic and structural scores
                source_sem_score = (
                    entity_names[rel["source"]].score
                    if rel["source"] in entity_names
                    else 0
                )
                target_sem_score = (
                    entity_names[rel["target"]].score
                    if rel["target"] in entity_names
                    else 0
                )
                # Get PageRank scores
                source_pr_score = pagerank_scores.get(rel["source"], 0)
                target_pr_score = pagerank_scores.get(rel["target"], 0)

                # Combine scores:
                # - Take max of semantic scores to promote paths with at least one relevant entity
                # - Take average of PageRank scores to consider both endpoints' importance
                # - Weight semantic score higher than PageRank (0.7 vs 0.3)
                semantic_score = (source_sem_score + target_sem_score) / 2
                structural_score = (source_pr_score + target_pr_score) / 2
                base_score = (
                    semantic_weight * semantic_score
                    + (1 - semantic_weight) * structural_score
                )

                rel_key = (rel["source"], rel["target"], rel["type"])
                if rel_key not in relationship_scores:
                    relationship_scores[rel_key] = []
                relationship_scores[rel_key].append((rel_item, base_score))
            rt.add_outputs({"relationship_scores": relationship_scores})
        async with ls.trace(
            "sort_relationships", inputs={"relationships": str(relationship_scores)}
        ):
            final_relationships = []
            for rel_list in relationship_scores.values():
                if not rel_list:
                    continue

                rel, score = rel_list[-1]
                if rel is None:
                    continue
                frequency_boost = len(rel_list)
                final_score = score * math.sqrt(frequency_boost)
                final_relationships.append((rel, final_score))

            final_relationships.sort(key=lambda x: x[1], reverse=True)
            top_relationships = final_relationships[:limit]
            rt.outputs.update({"top_relationships": str(top_relationships)})
        top_entities = sorted(
            [item for item in entity_names.values()],
            key=lambda x: x.score,
            reverse=True,
        )
        top_entities = top_entities[:limit]
        chunk_ids: list[tuple[str, str]] = sorted(
            {
                (rel.namespace[-1], rel.value["chunk_id"])
                for (rel, _) in top_relationships
            }
        )
        chunks = await asyncio.gather(
            *[
                store.aget(
                    ("graph", user_id, "chunks", chunk[0]),
                    key=str(chunk[1]),
                )
                for chunk in chunk_ids
            ]
        )

        return QueryResult(
            entities=top_entities,
            relationships=top_relationships,
            sources=[chunk for chunk in chunks if chunk is not None],
        )

    except Exception as e:
        logger.error(f"Error during graph query: {e}")
        raise


@ls.traceable
def compute_pagerank(
    relationships: List[Item],
    damping: float = 0.85,
    iterations: int = 30,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """Compute PageRank scores for entities in the graph with early stopping.

    Args:
        relationships: List of relationship items
        damping: Damping factor (default: 0.85)
        iterations: Maximum number of iterations (default: 30)
        tolerance: Convergence threshold for early stopping (default: 1e-6)
    """
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    entities = set()
    for rel_item in relationships:
        rel = rel_item.value
        source, target = rel["source"], rel["target"]
        entities.add(rel["source"])
        entities.add(rel["target"])
        outgoing[source].append(target)
        incoming[target].append(source)

    scores = {entity: 1.0 / len(entities) for entity in entities}
    rt = ls.get_current_run_tree() or ls.RunTree()

    for i in range(iterations):
        new_scores = {}
        for entity in entities:
            score = (1 - damping) / len(entities)

            if entity in incoming:
                for source in incoming[entity]:
                    num_outgoing = len(outgoing.get(source, []))
                    if num_outgoing > 0:
                        score += damping * scores[source] / num_outgoing

            new_scores[entity] = score

        total = sum(new_scores.values())
        new_scores = {k: v / total for k, v in new_scores.items()}

        max_diff = max(abs(new_scores[k] - scores[k]) for k in scores)
        scores = new_scores
        if max_diff < tolerance:
            rt.metadata["stop_reason"] = "converged"
            break
    else:
        rt.metadata["stop_reason"] = "max_iterations"

    return scores


def format_memories(memories: QueryResult) -> str:
    """Format query results into a readable memory format.

    Args:
        memories: Query results containing entities, relationships and sources

    Returns:
        Formatted string with relevant context from the knowledge graph
    """
    if not memories or (not memories["entities"] and not memories["relationships"]):
        return "No relevant memories found."

    sections = []

    # Format entities by type
    if memories["entities"]:
        sections.append("### Key Entities\n")
        # Group entities by type
        by_type = {}
        for entity in memories["entities"]:
            etype = entity.value["type"].lower()
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(entity.value)

        # Format each type group
        for etype, entities in by_type.items():
            sections.append(f"#### {etype.title()}")
            for e in entities:
                sections.append(f"- {e['name']}: {e['description']}")
            sections.append("")

    # Format relationships with context
    if memories["relationships"]:
        sections.append("### Key Relationships\n")
        for rel, score in memories["relationships"]:
            timestamp = rel.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            if rel.value.get("description"):
                sections.append(f"  {rel.value['description']} ({timestamp})")
            else:
                sections.append(
                    f"- {rel.value['source']} {rel.value['type'].lower()} {rel.value['target']} ({timestamp})"
                )
        sections.append("")

    # Add source context
    if memories["sources"]:
        sections.append("## Source Context\n")
        for i, source in enumerate(memories["sources"], 1):
            sections.append(f"{i}. {source.value['text']}")
        sections.append("")
    if not sections:
        return ""
    formatted = "\n".join(sections)
    return f"""

## Memories
You've recorded the following in previous conversations with the user:
<!-- start of memories -->
{formatted}
<!-- end of memories -->
"""
