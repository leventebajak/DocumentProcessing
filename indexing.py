import chromadb
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document, load_index_from_storage
from llama_index.core.node_parser import SimpleFileNodeParser, SentenceSplitter, HierarchicalNodeParser
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from settings import embed_model

db = chromadb.PersistentClient(path="chromadb")
chroma_collection = db.get_or_create_collection("rag")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

transformations = [
    SimpleFileNodeParser(),
    SentenceSplitter(chunk_size=100, chunk_overlap=10),
    HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
]


def _index_documents(documents: list[Document]):
    return VectorStoreIndex.from_documents(documents,
                                           transformations=transformations,
                                           storage_context=storage_context,
                                           embed_model=embed_model,
                                           show_progress=True)


def index_pdf(file_path: str):
    documents = PyMuPDFReader().load(file_path=Path(file_path))
    # The PDF reader creates a separate Document for each page by default.
    return _index_documents(documents)


def index_directory(directory_path: str):
    documents = SimpleDirectoryReader(directory_path).load_data()
    return _index_documents(documents)


def load_index():
    return VectorStoreIndex.from_vector_store(vector_store,
                                              transformations=transformations,
                                              storage_context=storage_context,
                                              embed_model=embed_model)


if __name__ == "__main__":
    # index = index_directory("data")
    index = index_pdf("..\\Docs\\1810.04805v2.pdf")
