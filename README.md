# Help Scout Assistant

## Overview

Help Scout Assistant is a document processing and query-response system that leverages Pinecone for vector storage and retrieval. The tool allows you to load PDF documents into a vector store, where they can be queried using OpenAI's language models. The retrieval process aims to simulate a customer support assistant by fetching relevant information from the document database in response to user queries.

## Features

- **PDF Document Processing**: Extracts text from PDFs and splits it into manageable chunks.
- **Vector Storage with Pinecone**: Stores text chunks in Pinecone for efficient retrieval.
- **Intelligent Query Handling**: Utilizes OpenAI's language model to generate human-like responses to user queries.
- **Customizable Settings**: The tool can be tailored through various environment variables for API keys and service configurations.

## Installation

### Prerequisites

- Python 3.7+
- A Pinecone account and API key
- An OpenAI account and API key
- Required Python packages as listed in `requirements.txt`

### Environment Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/hi-tech-AI/help-scout-assistant-using-Pinecone-vector-database.git
   cd help-scout-assistant-using-Pinecone-vector-database
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**

   Set up your environment variables using a `.env` file:

   ```plaintext
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

The primary interaction with this system happens through the `main` function, which coordinates the entire workflow of loading documents, creating/updating the index, and handling user queries.

```python
from your_script import main

user_query = "Your question here"
response = main(user_query)
print(response)
```

## Structure

- `extract_text_from_pdf`: Loads and splits the text from a given PDF path.
- `get_index_from_pinecone`: Ensures the specified index exists in Pinecone.
- `upsert_doc_to_pinecone`: Inserts or updates the document in the Pinecone vector store.
- `search_query`: Executes a search query on the document store and returns the response.
- `main`: Orchestrates the flow from PDF extraction to query response.

## Customization

You can modify the tool’s behavior by adjusting parameters in functions like `CharacterTextSplitter`, `ChatOpenAI`, and changing environment variable values in `.env`.

## Contributing

Feel free to open issues or submit pull requests to discuss and improve the functionality.
