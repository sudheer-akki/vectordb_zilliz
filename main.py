import os
import torch
from vector_db import VectorDatabase
from utils import convert_embeddings
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loading credential from .env file
Zilliz_CLUSTER_USER = os.getenv("USERNAME")
Zilliz_CLUSTER_PWD = os.getenv("PASSWORD")
URI = os.getenv("URI")
TOKEN = os.getenv("TOKEN")


vector_db = VectorDatabase(Zilliz_CLUSTER_USER=Zilliz_CLUSTER_USER,
                            Zilliz_CLUSTER_PWD=Zilliz_CLUSTER_PWD,
                            TOKEN= TOKEN,
                            URI=URI,
                            db_name="rag_demo",
                            collection_name="rag_collection",
                            vector_field_dim= 384,
                            metric_type = "COSINE")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=200,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

if __name__=="__main__":
    #loading both tokenizer and embedding model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # document with text data.
    with open("data.txt") as f:
        state_of_the_union = f.read()

    texts = text_splitter.create_documents([state_of_the_union])

    embeddings = convert_embeddings(text=texts,
                                    tokenizer=tokenizer,
                                    embed_model=embed_model,
                                    device=device)

    vector_db._insert_data(data=embeddings)


    query = "write your question"

    query_embeddings = convert_embeddings(text = query,
                                          tokenizer=tokenizer,
                                          embed_model=embed_model,
                                          device=device)

    output = vector_db._search_and_output_query(
                    query_embeddings= query_embeddings,
                    response_limit = 3)


    print(output) 