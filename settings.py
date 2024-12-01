from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

llm = Ollama(model="llama3.2", request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = llm
Settings.embed_model = embed_model
