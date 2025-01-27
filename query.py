from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from settings import llm
from indexing import load_index, get_retreiver

index = load_index()

query_str = input("Query: ")

print("\nGenerating response...")

retriever = get_retreiver(index, verbose=False)

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

print("Response: ")

streaming_response.print_response_stream()
