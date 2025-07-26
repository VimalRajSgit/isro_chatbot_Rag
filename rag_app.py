import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- Page Configuration ---
st.set_page_config(
    page_title="ISRO Documents Q&A",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Load Environment Variables ---
@st.cache_resource
def load_environment():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()
    return groq_api_key


# --- Initialize Models ---
@st.cache_resource
def initialize_models(groq_api_key):
    # Constants
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    LLM_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # Initialize embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Initialize LLM
    llm = ChatGroq(
        temperature=0,
        model_name=LLM_MODEL_NAME,
        api_key=groq_api_key
    )

    return embedding_function, llm


# --- Load Vector Database ---
@st.cache_resource
def load_vectorstore(_embedding_function):
    CHROMA_PERSIST_DIR = "chroma_db"

    if not os.path.exists(CHROMA_PERSIST_DIR):
        st.error(f"ChromaDB directory '{CHROMA_PERSIST_DIR}' not found. Please ensure the database exists.")
        st.stop()

    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=_embedding_function
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading ChromaDB: {str(e)}")
        st.stop()


# --- Create RAG Chain ---
@st.cache_resource
def create_rag_chain(_vectorstore, _llm):
    # Create MultiQuery Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        ),
        llm=_llm
    )

    # Define prompt template
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

    # Create RAG chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | _llm
            | StrOutputParser()
    )

    return rag_chain


# --- Main Application ---
def main():
    # Header
    st.title("🚀 ISRO Documents Q&A System")
    st.markdown("Ask questions about ISRO missions and get answers from your document database.")

    # Initialize components
    groq_api_key = load_environment()
    embedding_function, llm = initialize_models(groq_api_key)
    vectorstore = load_vectorstore(embedding_function)
    rag_chain = create_rag_chain(vectorstore, llm)

    # Sidebar with information
    with st.sidebar:
        st.header("📊 Database Info")

        # Get database statistics
        try:
            collection = vectorstore._collection
            doc_count = collection.count()
            st.metric("Total Documents", doc_count)
        except:
            st.info("Database loaded successfully")

        st.markdown("---")
        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - Be specific in your questions
        - Ask about ISRO missions, satellites, or launch vehicles
        - Try different phrasings if you don't get the expected answer
        """)

        st.markdown("---")
        st.markdown("### ⚙️ Model Information")
        st.markdown("""
        - **Embeddings**: BAAI/bge-large-en-v1.5
        - **LLM**: Llama-4 Maverick 17B
        - **Retrieval**: Multi-Query Retriever
        """)

    # Main chat interface
    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about ISRO missions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching through documents..."):
                try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
