# NutukGPT

A specialized RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) chatbot to provide accurate insights into Mustafa Kemal Atatürk's *Nutuk*. The system combines semantic vector search with web search capabilities to offer grounded historical context.

## Overview

This application parses a digitized version of *Nutuk*, utilizing a high-performance vector store for retrieval and the Model Context Protocol (MCP) to orchestrate external tools. 

## Design Decisions

This project has two main strategies to optimize performance for the Turkish language:

* **LLM-Based Re-Ranking:**
    Retrieval accuracy is boosted by a two-stage process. Initial chunks are retrieved via vector search, then passed through a lightweight LLM (using OpenAI Structured Outputs) to semantically re-rank them. This ensures the system prioritizes the most relevant chunks over those that merely share keywords.

* **Fine-Tuned Turkish Embeddings:**
    Instead of generic multilingual models, the system utilizes `selmanbaysan/turkish_embedding_model_fine_tuned`. This model is finetuned on Turkish datasets, improving semantic similarity matching for the unique linguistic features.
    
* **Linguistic-Aware Chunking:**
    Using `nltk`'s **Turkish sentence tokenizer** ensures that sentences containing titles common in the text (e.g., "Gen.", "Prof.") are not split incorrectly, preserving the semantic integrity of every vector.

## Tech Stack

* **LLM:** OpenAI (`gpt-5-nano`)
* **Embeddings:** Sentence Transformers 
* **Vector Database:** ChromaDB (Persistent storage)
* **Orchestration:** OpenAI Agents SDK 
* **Tools (MCP):** Brave Search
* **Interface:** Gradio
* **Validation:** Pydantic (Structured Outputs)
* **Processing:** `pypdf`, `nltk`
