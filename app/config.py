import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


load_dotenv()


CHAT_MODEL = AzureChatOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

)

EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
)

PINECONE_PATH = "./pinecone_db"