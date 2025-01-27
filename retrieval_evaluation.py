import asyncio
from random import sample
from llama_index.core.evaluation import RetrieverEvaluator, generate_question_context_pairs

from settings import llm
from indexing import load_index


async def main():
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=10)

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate", "recall", "precision"], retriever=retriever
    )

    nodes = sample(list(index.docstore.docs.values()), 10)

    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=3
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)

    eval_results = [result for result in eval_results if "Answer" not in result.query and
                    not result.query.startswith("Here are") and not result.query.startswith("(")]

    print("Evaluation results:")
    for eval_result in eval_results:
        print(eval_result)

    print(f"Hit rate: {sum([result.metric_dict['hit_rate'].score for result in eval_results]) / len(eval_results)}")
    print(f"MRR: {sum([result.metric_dict['mrr'].score for result in eval_results]) / len(eval_results)}")
    print(f"Recall: {sum([result.metric_dict['recall'].score for result in eval_results]) / len(eval_results)}")
    print(f"Precision: {sum([result.metric_dict['precision'].score for result in eval_results]) / len(eval_results)}")


if __name__ == "__main__":
    asyncio.run(main())
