from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import FaithfulnessEvaluator
# from llama_index.core.evaluation import RelevancyEvaluator

from settings import llm
from indexing import load_index, get_retreiver

index = load_index()
retriever = get_retreiver(index)

response_synthesizer = get_response_synthesizer(llm=llm)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

evaluator = FaithfulnessEvaluator(llm=llm)
# evaluator = RelevancyEvaluator(llm=llm)

query = input("Query: ")

response = query_engine.query(query)
print("Response:", response)

eval_result = evaluator.evaluate_response(response=response, query=query)

print(f"\nPassing: {eval_result.passing}\n")

print("Generated evaluation result:")
print(eval_result.feedback)
