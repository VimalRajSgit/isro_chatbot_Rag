import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever  # Import the advanced retriever

# --- 1. Load Environment Variables ---
# Load the API key from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- 2. Define Constants ---
# Use relative paths for portability
DATA_PATH = "satellite_Data"
CHROMA_PERSIST_DIR = "chroma_db"

# Using a top-tier embedding model for high-quality retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# Using the most powerful LLM on Groq for generation and query transformation
LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- 3. Initialize Models ---
# Use the upgraded, more powerful embedding model.
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},  # Use CPU for embeddings
    encode_kwargs={'normalize_embeddings': True}  # Recommended for BGE models
)

# Initialize the Groq LLM with the upgraded model
llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, api_key=groq_api_key)

# --- 4. Build or Load the Vector Database ---
# IMPORTANT: If you change the embedding model, delete the old 'chroma_db' folder.
if not os.path.exists(CHROMA_PERSIST_DIR):
    print(f"Persistent database not found. Creating a new one at '{CHROMA_PERSIST_DIR}'...")

    # Load documents from the specified directory
    print(f"Loading documents from '{DATA_PATH}'...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No .txt files found in '{DATA_PATH}'. Please check the path and file extensions.")

    # Using a larger chunk size to keep related information together.
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = text_splitter.split_documents(documents)

    # Create and persist the Chroma vector store
    print(f"Creating vector store with '{EMBEDDING_MODEL_NAME}' embeddings... (This may take a while)")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("✅ Vector store created successfully!")
else:
    print(f"Loading existing persistent database from '{CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )
    print("✅ Database loaded successfully!")

# --- 5. Create the RAG Chain with an Advanced Retriever ---
print("Creating RAG chain with MultiQueryRetriever...")

# === ACCURACY IMPROVEMENT: USE MULTI-QUERY RETRIEVER ===
# This uses the LLM to generate multiple queries from different perspectives
# for a given user question. This is a powerful technique for improving retrieval accuracy.
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Define the prompt template
prompt_template = """
You are a highly specialized assistant for answering questions about ISRO missions. Your answers must be precise and based *only* on the context provided.
Do not add any information that is not explicitly stated in the context. If the answer cannot be found in the given context, you must state: "The provided context does not contain the answer to this question."

CONTEXT:
{context}

QUESTION:
{question}

PRECISE ANSWER:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create the RAG chain using LangChain Expression Language (LCEL)
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print("✅ RAG system is ready. You can now ask questions.")

# --- 6. Query the System ---
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question about the documents (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        print("\nThinking...")
        answer = rag_chain.invoke(query)
        print("\nAnswer:", answer)