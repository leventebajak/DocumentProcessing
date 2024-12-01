from llama_index.core.chat_engine.types import ChatMode

from settings import llm
from indexing import load_index

print("Loading index from storage...")
index = load_index()

chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, llm=llm)

exit_command = "exit"
print(f'Type "{exit_command}" to exit.\n')

message = input("You: ")
while message != exit_command:
    streaming_response = chat_engine.stream_chat(message)
    print("Bot: ", end="", flush=True)
    streaming_response.print_response_stream()
    print()
    message = input("You: ")

# You: What is BERT? Also, which file mentions it?
# Bot: BERT stands for Bidirectional Encoder Representations from Transformers. It's a type of deep learning model designed to pre-train deep bidirectional representations from unlabeled text.
# The file that mentions BERT is the PDF document located at: `D:\...\1810.04805v2.pdf`. Specifically, it's mentioned in a citation: "Rad- ford et al. 2018)".
