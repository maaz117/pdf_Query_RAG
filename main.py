from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from pyngrok import ngrok
import nest_asyncio

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt


import torch

# Define the query model for FastAPI
class Query(BaseModel):
    query_str: str

# Initialize the FastAPI app
app = FastAPI()

from transformers import BitsAndBytesConfig

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt="You are a Q&A assistant...",
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    model_name="HuggingFaceH4/zephyr-7b-beta",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)


documents=SimpleDirectoryReader('E:/project02/data').load_data()
documents

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")) # Directly using the embedding model name here for simplicity

# Initialize the ServiceContext and VectorStoreIndex
service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

service_context

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

@app.get("/")
async def welcome():
    """ Redirects to the API documentation """
    return RedirectResponse(url='/docs')

@app.post("/query/")
async def handle_query(query: Query):
    """ Endpoint to handle PDF content queries """
    try:
        response = query_engine.query(query.query_str)
        return {"response": response}  # Simplify JSON response creation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app using Uvicorn and ngrok
if __name__ == "__main__":
    # Setup ngrok
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)

    # Allow nested asyncio calls
    nest_asyncio.apply()

    # Run the app with Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)