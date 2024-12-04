## GraphRAG Template

A LangGraph template implementing knowledge-graph based Retrieval-Augmented Generation (Graph RAG) based on the [Hippo RAG paper](https://arxiv.org/abs/2405.14831).

## Overview

This template demonstrates how to build a chatbot with a graph-based memory system. Instead of storing memories as simple text chunks, it extracts entities and relationships to form a knowledge graph that can be queried using a combination of semantic search and topic-sensitive PageRank ([paper](https://dl.acm.org/doi/10.1145/511446.511513)).

## Why GraphRAG?

Traditional RAG systems treat each document or chunk independently or only contextualize within an existing document structure, which means multi-hop reasoning requires sequential, expanding searches to find related information. While vector/hybrid search excels at direct matches, it faces challenges with:

- Multi-hop reasoning: Finding connections across documents requires multiple sequential searches
- Reasoning about global context: The agent would have to query out over the entire raw corpus which creates expensive and slow query-time overhead.

GraphRAG, if well-tuned to your use case, can address these limitations by:

- Enabling single-step multi-hop retrieval instead of requiring sequential searches
- Enabling graph aggregations over entities and relationships to reason about global context

The PageRank algorithm helps focus the search on the most important nodes in the graph, which the paper shows can improve retrieval performance.

Note: the global aggregations are not currently implemented in this template, since they are not included in the Hippo RAG paper.

## Quick Start

1. Create a `.env` file:

```bash
cp .env.example .env
```

2. Set API keys:

```
ANTHROPIC_API_KEY=your-api-key  # For Anthropic
OPENAI_API_KEY=your-api-key     # For OpenAI
```

3. Install dependencies:

```bash
pip install -e .
pip install -U "langgraph-cli[inmem]
```

4. Run the template:

```bash
langgraph dev
```

## How It Works

### Entity Extraction

The system automatically extracts entities from conversations and documents:

- **Name**: Unique identifier (e.g., "John Smith", "Project X")
- **Type**: Category (e.g., person, location, organization, concept)
- **Description**: Brief context about the entity

You can customize entity extraction by modifying the system prompt in `src/graphrag/prompts.py`.

### Relationship Building

Relationships between entities are identified to form a knowledge graph:

- **Source**: Starting entity
- **Target**: Ending entity
- **Type**: Relationship type (e.g., "works_for", "located_in")
- **Description**: Additional context about the relationship

The relationship extraction can be tuned by adjusting the prompt guidelines in `SYSTEM_PROMPT`.

### Memory Retrieval

When the chatbot needs to recall information, it:

1. Performs semantic search to find relevant entities
2. Uses PageRank to identify important nodes
3. Combines semantic + structural scores for optimal retrieval
4. Returns a ranked list of memories with source context

The retrieval process can be customized by:

- Adjusting the PageRank parameters in `src/chatbot/utils.py`
- Modifying the scoring weights between semantic and structural signals
- Changing the chunk size and overlap for document processing

### Efficient Updates

The system uses debouncing to batch memory updates:

- Schedules updates after periods of inactivity
- Cancels pending updates if new messages arrive
- Reduces processing overhead and costs

## Customization Guide

### Modifying Entity & Relationship Extraction

The extraction behavior is controlled by the system prompt in `src/graphrag/prompts.py`. You can:

- Add new entity types by updating the entity extraction guidelines
- Change relationship extraction rules by modifying the relationship guidelines
- Adjust the extraction threshold by updating the content grounding section

Example prompt modification:

```python
# In src/graphrag/prompts.py
SYSTEM_PROMPT = """
... modify guidelines for entity/relationship extraction ...
"""
```

### Tuning Memory Retrieval

The retrieval process can be customized by using different configuration parameters.

```python
# Adjust chunk size for document processing
chunk_size = config["configurable"].get("chunk_size", 12_000)

# Modify chunk overlap
chunk_overlap = config["configurable"].get("chunk_overlap", 0.25)

### Adding Custom Functionality
The graph-based architecture makes it easy to add new features:
1. Add new node types by creating new Pydantic models
2. Extend the graph processing by adding new nodes to the StateGraph
3. Customize the storage backend by modifying the BaseStore implementation

## Try It Out

Open this template in LangGraph Studio to get started. Chat with the bot and watch as it builds a knowledge graph from your conversation. The graph visualization will show you how information is being connected and stored.

## Contributing

We welcome contributions! Some areas that could use improvement:
- Additional entity and relationship types
- Better extraction prompts
- More efficient graph algorithms
- New retrieval strategies
- Improved visualization tools

Please submit pull requests or open issues for any improvements.


<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "chatbot": {
      "type": "object",
      "properties": {}
    },
    "memory_graph": {
      "type": "object",
      "properties": {
        "model": {
          "type": "string",
          "default": "anthropic:claude-3-5-sonnet-20240620",
          "description": "The name of the language model to use for the agent. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic:claude-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-2.0",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-2.1",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-3-5-sonnet-20240620",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-3-haiku-20240307",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-3-opus-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-3-sonnet-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic:claude-instant-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai:gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      }
    }
  }
}
```

-->
