import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

_vectorstore = None
_retriever   = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX", "research-agent"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    return _vectorstore


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20},
        )
    return _retriever


def run_retrieval(query: str) -> str:
    try:
        docs = get_retriever().invoke(query)
        if not docs:
            return "No relevant documents found in knowledge base."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            title  = doc.metadata.get("title",  "untitled")
            topic  = doc.metadata.get("topic",  "unknown")
            formatted.append(
                f"[{i}] {title}\n"
                f"Source: {source}\n"
                f"Topic:  {topic}\n"
                f"{doc.page_content}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Retrieval failed: {str(e)}"
