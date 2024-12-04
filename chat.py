from llama_index.core.chat_engine import ContextChatEngine

from settings import llm
from indexing import load_index, get_retreiver

index = load_index()
retriever = get_retreiver(index)

chat_engine = ContextChatEngine.from_defaults(retriever=retriever, llm=llm)

exit_command = "exit"
print(f'Type "{exit_command}" to exit.\n')

message = input("You: ")
while message != exit_command:
    streaming_response = chat_engine.stream_chat(message)
    print("Bot: ", end="", flush=True)
    streaming_response.print_response_stream()
    print()
    message = input("\nYou: ")
