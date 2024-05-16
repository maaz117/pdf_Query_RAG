# Multiple Pdf Query RAG
#### Using Hugging Face LLM, LangChain, LlamaIndex and Docker

## Overview
This document provides the setup and operation instructions for an advanced API built using FastAPI, which integrates Hugging Face’s large language models (LLM) for sophisticated language processing, LlamaIndex for robust data indexing, and LangChain for powerful embeddings. This Retrieval-Augmented Generation (RAG) setup allows the API to efficiently query across multiple PDF documents, leveraging the combined strengths of contextual understanding and information retrieval. The application is containerized using Docker to facilitate easy deployment and scalability.

## Dependencies
This project relies on the following technologies:

- **FastAPI:** A modern web framework for building APIs.
- **Hugging Face LLM:** Utilizes large language models for natural language understanding and generation.
- **LlamaIndex:** For robust indexing and querying capabilities.
- **LangChain:** For leveraging powerful embeddings in information retrieval.
- **Pyngrok:** To expose local servers to public URLs.
- **Nest_asyncio:** Facilitates running asyncio nested within other asyncio loops.
- **Pydantic:** For data validation and settings management.
- **BitsAndBytes:** For memory-efficient model quantization, requires a *GPU*.

## Model Used
- **Hugging Face LLM:** Utilizes models like *zephyr-7b-beta* for response generation based on document content.
- **LangChain with Sentence Transformers:** For generating document embeddings using models like *sentence-transformers/all-mpnet-base-v2*.

## Installation
To set up and run the API locally:

1. Clone this repository to your local machine.
2. Ensure Docker and a GPU (for BitsAndBytes) are installed on    your system.
3. Build the Docker container using the provided Dockerfile.
4. Run the Docker container.

## Building and Running the Docker Container

To construct and launch the Docker container:

1. Navigate to the directory containing the FastAPI application.
2. Build the Docker image using the command
```
docker build -t project_api .
```
3. Launch the Docker container:
```
docker run -p 8000:8000 project_api
```

## Accessing the API
1. Access the API using the URL http://localhost:8000.
2. Explore the API documentation at http://localhost:8000/docs or http://localhost:8000/redoc.

## API Endpoints

- **GET /:** A welcome endpoint that redirects to the Swagger UI documentation.
- **POST /query/:** Endpoint to handle queries about PDF content. Accepts JSON input with a query_str field and returns query results.

## Interacting with the API
You can interact with the API using the Swagger UI or by making HTTP requests with tools like curl or Postman.

## License
This project is licensed under the Apache-2.0 License.
