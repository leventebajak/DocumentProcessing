# DocumentProcessing

This repository contains a set of Python scripts for document processing and querying.
You can easily install the required dependencies and start using the provided tools for indexing, querying, and chatting with an LLM.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Files and Directories Indexing

Running the `indexing.py` script from the terminal allows you to index files and directories, making them ready for search and retrieval.

## Querying the Index

Once the documents are indexed, you can use `query.py` to make a single query to retrieve an answer generated from the retrieved data.

## Chatting with the LLM

For a more interactive experience, use `chat.py` to start a chat session with the LLM. You can ask questions or interact with the model in a conversational way.

## Scripts Overview

- `indexing.py`: Indexes files and directories for search and retrieval.
- `query.py`: Allows you to query the indexed data.
- `chat.py`: Enables you to start a chat with the LLM.
- `response_evaluation.py`: Evaluates responses to queries.
- `retrieval_evaluation.py`: Evaluates the retrieval process.
- `settings.py`: Contains the model configurations.