# Local RAG-Based PDF Question Answering

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions using information retrieved from a nutrition textbook. Built with Python, PyTorch, Sentence Transformers, and Hugging Face Transformers.

The project demonstrates how to build a RAG workflow from scratch, covering document preprocessing, text chunking, embedding generation, semantic search, prompt augmentation, and local LLM inference.

## Overview

Large Language Models (LLMs) can generate useful responses, but their answers are not always grounded in a specific source of information.

This project uses Retrieval-Augmented Generation (RAG) to provide an LLM with relevant passages from a document before generating an answer.

The pipeline uses *Human Nutrition: 2020 Edition* as its knowledge source and runs the embedding, retrieval, and generation workflow locally.

## How It Works

The project follows three main stages:

### 1. Document Processing and Embedding Generation

- Downloads and extracts text from the nutrition textbook PDF.
- Processes the extracted text and splits it into sentences using spaCy.
- Groups sentences into chunks of 10.
- Filters out chunks containing 30 tokens or fewer.
- Generates text embeddings using the `all-mpnet-base-v2` Sentence Transformer model.
- Saves the processed text chunks and embeddings to a CSV file.

**Processing results:**
- PDF pages processed: 1,208
- Text chunks generated after filtering: 1,685
- Embedding dimensions: 768

### 2. Semantic Search and Retrieval

- Converts the user's query into an embedding using the same Sentence Transformer model.
- Calculates similarity scores between the query embedding and document embeddings using PyTorch dot-product similarity.
- Retrieves the top 5 most relevant text chunks.
- Retains the associated page numbers for source context.

### 3. Context Augmentation and Answer Generation

- Loads the Qwen2.5-0.5B-Instruct model locally using Hugging Face Transformers.
- Formats the retrieved passages into a prompt alongside the user's question.
- Uses the model's chat template to structure the prompt.
- Generates an answer using the retrieved document context.
- Provides an `ask()` function that combines retrieval, prompt construction, and generation.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Embedding Model | Sentence Transformers (`all-mpnet-base-v2`) |
| Language Model | Qwen2.5-0.5B-Instruct |
| NLP | spaCy |
| LLM Framework | Hugging Face Transformers |
| Data Processing | Pandas, NumPy |
| Development | Jupyter Notebook |
| Hardware Acceleration | Apple Silicon MPS |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/local_RAG.git
cd local_RAG
```

Install the required libraries:

```bash
pip install torch transformers sentence-transformers
pip install pandas numpy spacy tqdm requests pymupdf
```

Download the spaCy English pipeline is not required; this project uses spaCy's built-in English sentencizer.

## Running the Project

1. Open `RAG_notebook.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells in sequence.
3. The notebook downloads the nutrition textbook PDF if it is not already present.
4. Text is extracted, processed, chunked, and converted into embeddings.
5. The embeddings are saved as `text_chunks_and_embeddings_df.csv`.
6. The semantic retrieval and local LLM generation stages are executed.
7. Use the final `ask()` function to submit questions.

Example:

```python
query = "What are the macronutrients and what roles do they play?"

answer = ask(query=query)

print(answer)
```

## Hardware and Model Execution

The notebook includes support for Apple Silicon GPU acceleration through PyTorch MPS.

The Qwen2.5-0.5B-Instruct model is loaded locally using FP16 precision and PyTorch SDPA attention.

The notebook does not use BitsAndBytes 4-bit quantization because it is not supported by the MPS configuration used here.

## Project Scope

This project is a learning implementation of a local RAG pipeline. It demonstrates the core retrieval and generation workflow without relying on external LLM APIs or a dedicated vector database.

The embeddings are stored in CSV format, and semantic retrieval is implemented directly using PyTorch.

## Future Improvements

- Experiment with different chunk sizes and retrieval strategies.
- Implement a reranking model to improve retrieved passage ordering.
- Explore vector indexing for larger document collections.
- Improve source-page references in generated answers.
- Evaluate retrieval relevance and answer quality across a larger question set.

## Acknowledgements

- [Human Nutrition: 2020 Edition – University of Hawaiʻi](https://pressbooks.oer.hawaii.edu/humannutrition2/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PyTorch](https://pytorch.org/)
