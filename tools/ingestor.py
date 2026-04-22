import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

_embeddings = None
_vectorstore = None


def get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX", "research-agent"),
            embedding=_embeddings,
            # No namespace — everything in one flat index
        )
    return _vectorstore


def ingest_articles(articles: list[dict], topic: str) -> int:
    """
    Chunk and store articles.
    Topic stored as metadata — used for filtering and display.
    No namespace — flat index, cross-topic retrieval works cleanly.
    """
    if not articles:
        return 0

    docs = []
    for article in articles:
        if not article["full_text"]:
            continue
        chunks = _splitter.split_text(article["full_text"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": article["url"],
                    "title":  article["title"],
                    "topic":  topic,
                }
            ))

    if not docs:
        return 0

    get_vectorstore().add_documents(docs)
    print(f"[ingestor] Stored {len(docs)} chunks for topic: '{topic}'")
    return len(docs)
