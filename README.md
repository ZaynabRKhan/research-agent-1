# Research Agent

An AI-powered research assistant that automatically searches academic literature, retrieves relevant papers, extracts evidence, and generates evidence-grounded research reports using Retrieval-Augmented Generation (RAG).

Unlike traditional LLM-based summarization, the system retrieves relevant scientific literature, builds a vector database, and generates reports grounded only in the retrieved evidence.

---

## Features

- Generates multiple search queries from a research question
- Retrieves academic papers using the OpenAlex API
- Downloads PDFs when available and reconstructs abstracts otherwise
- Parses and chunks research papers
- Creates semantic embeddings and a Chroma vector database
- Retrieves relevant evidence using semantic search
- Generates structured research reports
- Critiques reports for faithfulness, completeness, and unsupported claims
- Automatically revises reports based on critic feedback
- Includes a benchmarking framework for evaluating different RAG configurations

---

## Pipeline

```
User Question
      │
      ▼
Generate Search Queries
      │
      ▼
Retrieve Papers (OpenAlex)
      │
      ▼
Download PDFs / Extract Abstracts
      │
      ▼
Parse & Chunk Documents
      │
      ▼
Generate Embeddings
      │
      ▼
Chroma Vector Database
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Report Generation
      │
      ▼
Critic Evaluation
      │
      ├───────────────┐
      │               │
  APPROVE         REVISE
      │               │
      │         Report Editor
      │               │
      └──────► Final Report
```

---

## Technologies

### AI & NLP

- Large Language Models
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Prompt Engineering

### Frameworks

- LangGraph
- LangChain
- ChromaDB

### Data Sources

- OpenAlex API

---

## Evaluation Framework

The project includes an automated benchmarking system for comparing different pipeline configurations.

The following parameters can be evaluated:

- Number of search queries
- Papers retrieved per query
- Chunk size
- Chunk overlap
- Retrieval strategy
- Relevance ranking
- Number of critic iterations
- Language model

Results are exported to CSV for comparison.

---

## Example Questions

- Does coffee cause blindness?
- What causes schizophrenia?
- Can LLMs replace software developers?
- What is the significance of CRISPR?
- Does intermittent fasting increase lifespan?
- How do bees use Earth's magnetic field?

---

## Future Improvements

- Support additional academic databases (PubMed, arXiv, Semantic Scholar)
- Hybrid retrieval using citation graphs
- Knowledge graph construction from retrieved literature
- Multi-agent research workflows
- Improved evidence ranking
- Interactive web interface

---

## Disclaimer

The generated reports are intended to assist literature exploration and should not be considered a substitute for expert review. Conclusions are limited to the evidence retrieved by the system.
