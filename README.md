🌌 Unified ISRO Intelligence Assistant
This repository contains a powerful, multi-functional AI assistant built with Streamlit, LangChain, and Groq. The assistant serves as a one-stop interface for interacting with various space and earth-observation data, focusing on the Indian Space Research Organisation (ISRO).

The primary application, Unified Intelligence Assistant, uses a sophisticated LLM-based router to intelligently delegate user queries to the appropriate tool:

ISRO Q&A: A Retrieval-Augmented Generation (RAG) system to answer questions about ISRO missions, technologies, and history.

Satellite Tracking: Real-time tracking of active ISRO satellites.

Environmental Analysis: Agricultural and environmental data retrieval (weather, flood risk) using NASA's POWER API.

The repository also includes a standalone version of the ISRO Documents Q&A System for users who only need the RAG functionality.

✨ Features
🧠 Intelligent Intent Routing: Uses a Llama 3 model via Groq to analyze user prompts and determine whether to answer a question, track a satellite, or fetch environmental data.

📚 Conversational ISRO Q&A: Ask complex questions about ISRO missions like "What was the payload of Chandrayaan-3?" or "Explain the PSLV rocket." The system uses a ChromaDB vector store and a MultiQueryRetriever to find the most relevant information from official documents.

🛰️ Live Satellite Tracking: Get the real-time latitude, longitude, and altitude of active ISRO satellites like Cartosat-3, Aditya-L1, or NVS-01. It also gracefully handles queries for defunct missions like Mangalyaan or Chandrayaan-1.

🌦️ Environmental & Agricultural Analysis: Request weather reports, monthly summaries, or flood risk assessments for any location. For example: "What was the weather in Bengaluru for the last 30 days?" or "Assess the flood risk for Chennai in June 2025."

⚡ High-Performance LLMs: Powered by the ultra-fast Groq API for near-instantaneous LLM responses.

🖥️ Interactive UI: A user-friendly chat interface built with Streamlit.

🛠️ Tech Stack & Architecture
The system's core is the Unified Intelligence Assistant, which acts as a router. When a user sends a query, it's first sent to the get_intent function. This function uses a Groq LLM to classify the query and extract necessary parameters, outputting a JSON object. Based on the intent, the system calls the appropriate handler function.

handle_qna: Activates the RAG chain built with LangChain. It retrieves context from a ChromaDB vector database (populated with ISRO documents) and generates a precise answer.

handle_satellite_tracking: Fetches Two-Line Element (TLE) data from CelesTrak and uses the skyfield library to calculate the satellite's current ground position.

handle_agri_analysis: Uses an LLM to geocode the user's location, then calls the NASA POWER API to get daily temporal data for analysis.

Frontend: Streamlit

LLM Inference: Groq (Llama 3, Llama 4 Maverick)

Orchestration: LangChain

Vector Database: ChromaDB

Embeddings: BAAI/bge-large-en-v1.5

Data Sources:

ISRO Documents (Official Site, MOSDAC, etc.) for RAG

NASA POWER API for environmental data

CelesTrak for satellite TLE data

🚀 Setup and Installation
Follow these steps to get the project running locally.

1. Prerequisites
Python 3.9 or higher

Git

2. Clone the Repository
Bash

git clone <your-repository-url>
cd <your-repository-directory>
3. Set Up a Virtual Environment
It's highly recommended to use a virtual environment.

Bash

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
(Note: You will need to create a requirements.txt file from the imports in the Python scripts.)

Bash

pip install -r requirements.txt
5. Set Up Environment Variables
Create a file named .env in the root directory and add your API keys.

GROQ_API_KEY="gsk_YourGroqApiKey"
NASA_API_KEY="YourNasaApiKey" # Or leave as DEMO_KEY, but it's rate-limited
6. Create the Vector Database
This is a critical step for the Q&A functionality. The RAG system relies on a local ChromaDB vector database populated with data from ISRO's official websites and data portals (like MOSDAC).

You will need to run your own data ingestion script (not included) to scrape, chunk, and embed the documents into a ChromaDB instance. The database should be persisted in a directory named chroma_db in the root of the project.

🏃‍♀️ Running the Application
You can run either the main unified assistant or the standalone Q&A system.

To Run the Full Unified Intelligence Assistant:
Bash

streamlit run unified_intelligence_assistant.py
To Run the Standalone ISRO Q&A System:
Bash

streamlit run isro_qna_system.py
Once the app is running, open your web browser to the local URL provided by Streamlit (usually http://localhost:8501) and start asking questions!

Example Prompts to Try:
What is the purpose of the Aditya-L1 mission?

Track SCATSAT-1

Give me a weather summary for Hyderabad for last month

What is the flood risk in Mumbai for the last 30 days?
