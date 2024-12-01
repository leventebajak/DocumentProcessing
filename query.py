from llama_index.core import get_response_synthesizer
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from settings import llm
from indexing import load_index

print("Loading index from storage...")
index = load_index()

query_str = input("Query: ")

print("Generating response...")

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    llm=llm,
    streaming=True,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

streaming_response = query_engine.query(query_str)
streaming_response.print_response_stream()
