
import logging.handlers
import nest_asyncio
import logging
nest_asyncio.apply()

handler = logging.handlers.RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)



from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.core import PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from GraphRAGImplementation.GraphRAGExtractor import GraphRAGExtractor
from GraphRAGImplementation.GraphRAGStore import GraphRAGStore
from GraphRAGImplementation.GraphRAGEngine import GraphRAGQueryEngine


from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

logger.info("Creating Models")
embedding_model = AzureOpenAIEmbedding(
    deployment_name="soti-openai-prod-datascience-sotigpt-ada02", 
    api_key="1f65cf0767b44c1ca4b33197c1fbd2ae", 
    azure_endpoint="https://soti-openai-prod-datascience.openai.azure.com/",
    model="text-embedding-ada-002",
    api_version="2024-02-01"
)
llm = AzureOpenAI(
    deployment_name="soti-openai-prod-datascience-sotigpt-4omini", 
    api_key="1f65cf0767b44c1ca4b33197c1fbd2ae", 
    azure_endpoint="https://soti-openai-prod-datascience.openai.azure.com/",
    model="gpt-4o-mini",
    api_version="2024-02-01"
)
Settings.embed_model = embedding_model
Settings.llm = llm


# llm=Ollama(model="deepseek-r1:1.5b", base_url=f"http://localhost:11434", 
#            request_timeout=900.0, keep_alive="30m")
# Settings.llm = llm
# embed_model = OllamaEmbedding(model_name="bge-m3", 
#                               base_url=f"http://localhost:11434", 
#                               trust_remote_code=True)
# Settings.embed_model = embed_model



logger.info("Creating Nodes")
loader = SimpleDirectoryReader(
            input_dir = "input-dir",
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(docs)

graph_store = GraphRAGStore(
    llm = llm,
    username="neo4j", password="neo4JPassword", url="bolt://localhost:7687"
)

kg_extractor = GraphRAGExtractor(
    llm=llm,
    max_paths_per_chunk=25,
)

index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

print(index.property_graph_store.get_triplets()[:3])



