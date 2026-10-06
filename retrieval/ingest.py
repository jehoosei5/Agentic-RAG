from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.config import (
    EMBEDDING_MODEL
)


DOCS_PATH = Path("docs")


def load_documents():
    documents = []

    for file_path in DOCS_PATH.rglob("*.txt"):

        source = file_path.parent.name
        filename = file_path.name

        text = file_path.read_text(
            encoding="utf-8"
        )

        document = Document(
            page_content=text,
            metadata={
                "source": source,
                "document": filename,
                "path": str(file_path),
            }
        )

        documents.append(document)

    return documents


def create_vector_store():

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="agentic-rag",
    )

    print("Vector store created.")

    return vectorstore