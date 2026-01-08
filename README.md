# Aetheria: Elara the NPC

Aetheria is a project demonstrating an AI-powered NPC (Non-Player Character) named Elara, built using LangChain, LangGraph, and LM Studio. The system features a RAG (Retrieval-Augmented Generation) pipeline for lore faithfulness and a sentiment analysis loop for relationship management.

## Project Structure

- `app.py`: Streamlit-based UI for interacting with Elara.
- `src/agent.py`: LangGraph logic for the NPC brain and sentiment analysis.
- `src/database.py`: Handles lore document loading and vector store (ChromaDB) management.
- `src/evaluator.py`: Contains the `Judge` class for real-time and benchmark evaluation.
- `src/benchmark.py`: Automation script to evaluate the pipeline against a dataset.
- `world_lore/`: Markdown files containing the game's lore.
- `benchmark_data.json`: A dataset of 20 questions and expected lore facts for testing.

## Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running with a loaded model (e.g., Llama 3.2 3B).
- Local server started on port 1234.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Digitize the lore (Initialize the Vector Database):
   ```bash
   python src/database.py
   ```

## Usage

### Run the Interactive App
```bash
streamlit run app.py
```

### Run the Performance Benchmark
To evaluate the pipeline's accuracy and faithfulness against the 20 test cases:
```bash
python -m src.benchmark
```
This will generate a `benchmark_results.json` file and display the performance scores.

## Evaluation Pipeline

The project uses the LLM-as-a-Judge method (like the evaluators offered by LangChain) to grade the NPC's performance. A separate LLM instance acts as an impartial judge, analyzing the interaction to calculate the following four RAG metrics:

- **Context Precision**: Did the AI find the right documents?
   - The Judge compares the `user input` against the `retrieved documents`.
   - If this metric is low, adjust **embeddings** and/or document **chunk size**.
- **Faithfulness (Hallucination)**: Did the LLM make things up that are not found in the documents?
   - The Judge verifies if every claim in the NPC's response is supported by the retrieved context.
   - If this metric is low, lower `temperature` or refine `prompt`.
- **Answer Relevance:** Did the LLM ignore the question of the user?
   - The Judge ensures the response addresses the prompt rather than off-topic drift. 
   - If this metric is low, system prompt might be too restrictive.
- **Correctness:** Did the LLM get the facts right compred to the benchmark (gold-standard) question/answer dataset?
   - The Judge compares the NPC's response against a hand crafted dataset (`benchmark_data.json`) to check for semantic accuracy.

Metric,Score,Description
Avg Faithfulness (No Hallucination),0.78,Did the LLM make things up not found in the documents?
Avg Context Precision (Retrieval),0.59,Did the AI find the right documents?
Avg Answer Relevance (Coherence),0.86,Did the LLM actually answer the user's question?
Avg Correctness (Ground Truth),0.75,Did the LLM get the facts right compared to the gold standard?
