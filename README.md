# Research Agent

An agentic research assistant for querying and comparing academic papers.

The app uses Gemini, Chroma, and retrieval-augmented generation to answer questions from uploaded PDFs. Instead of using a fixed retrieval flow, Gemini can choose which retrieval tool to use depending on the question.

## Features

* Upload and query multiple academic PDFs
* Semantic search with Gemini embeddings and Chroma
* Agentic retrieval using Gemini function calling
* Search within a single paper or across multiple papers
* Retrieve additional page context when needed
* Page-level citations
* Visible agent activity showing which tools were used
* Deployed on Google Cloud Run

## How it works

Uploaded PDFs are parsed with PyMuPDF and split into overlapping text chunks. Each chunk is embedded with Gemini and stored in Chroma together with its source metadata.

When a question is submitted, Gemini can choose between three retrieval tools:

* `search_one_paper` for questions about a specific paper
* `search_all_papers` for cross-paper questions or comparisons
* `get_page_context` when more surrounding evidence is needed

The retrieved passages are returned to Gemini, which produces a grounded answer with references to the relevant paper and page.

## Tech Stack

Python · Streamlit · Gemini API · ChromaDB · PyMuPDF · Google Cloud Run

## Demo

[**Live demo**](https://research-assistant-585244727044.europe-west1.run.app/)
