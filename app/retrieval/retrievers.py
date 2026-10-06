from langchain_pinecone import PineconeVectorStore

from app.config import (
    EMBEDDING_MODEL,
)

vectorstore = PineconeVectorStore.from_existing_index(
        embedding=EMBEDDING_MODEL,
        index_name="agentic-rag",
    )



def get_retriever(source: str):

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {
                "source": source
            }
        }
    )


def retrieve_documents(
    question: str,
    source: str,
):

    retriever = get_retriever(source)

    documents = retriever.invoke(
        question
    )

    return documents