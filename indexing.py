import chromadb
from pathlib import Path
from httpx import ConnectError
from argparse import ArgumentParser
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from settings import embed_model

path = "chromadb"
docstore_path = f"{path}/docstore.json"

db = chromadb.PersistentClient(path=path)
chroma_collection = db.get_or_create_collection("rag")

pymupdf_reader = PyMuPDFReader()
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
try:
    docstore = SimpleDocumentStore.from_persist_path(docstore_path)
except FileNotFoundError:
    docstore = SimpleDocumentStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)

transformations = [
    SentenceSplitter.from_defaults(),
    HierarchicalNodeParser.from_defaults(),
]


def _index_documents(documents: list[Document]):
    try:
        for doc in documents:
            doc.metadata["page"] = doc.metadata.pop("source")
        index = VectorStoreIndex.from_documents(
            documents=documents,
            transformations=transformations,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
            store_nodes_override=True,
        )
        index.docstore.persist(docstore_path)
        return index
    except ConnectError:
        raise ConnectError("The Ollama server is not running. Please start the server and try again.")


def index_pdf(file_path: str):
    documents = pymupdf_reader.load(file_path=Path(file_path))
    # The PDF reader creates a separate Document for each page by default.
    return _index_documents(documents)


def index_directory(directory_path: str):
    documents = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".pdf"],
        file_extractor={".pdf": pymupdf_reader},
    ).load_data()
    return _index_documents(documents)


def load_index():
    try:
        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            transformations=transformations,
            embed_model=embed_model,
        )
    except ConnectError:
        raise ConnectError("The Ollama server is not running. Please start the server and try again.")


def get_retreiver(index: VectorStoreIndex, similarity_top_k=10, verbose=False):
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k, verbose=verbose)
    retriever = AutoMergingRetriever(base_retriever, storage_context=index.storage_context, verbose=verbose)
    return retriever


if __name__ == "__main__":
    parser = ArgumentParser(description='Index a document or directory of documents.')
    parser.add_argument('path', type=str, help='The path to the document or directory to index.')
    args = parser.parse_args()

    path = args.path
    if Path(path).is_file():
        if Path(path).suffix != '.pdf':
            parser.error(f'The document {path} is not a PDF.')
        index = index_pdf(path)
    elif Path(path).is_dir():
        index = index_directory(path)
    else:
        parser.error(f'The path {path} does not exist.')
