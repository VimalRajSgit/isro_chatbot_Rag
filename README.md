# 🌌 ISRO Intelligence Assistant

A comprehensive Streamlit-based application that provides real-time satellite tracking, ISRO mission Q&A, and environmental analysis capabilities. This project integrates multiple data sources including official ISRO documentation, NASA environmental data, and live satellite telemetry.

## ✨ Features

### 🛰️ Real-Time Satellite Tracking
- Live position tracking for 25+ active ISRO satellites
- TLE (Two-Line Element) data from CelesTrak
- Interactive position display with coordinates and altitude
- Automatic satellite name matching with fuzzy search
- Status handling for defunct missions (Chandrayaan-1, Chandrayaan-2, Mangalyaan)

### 🚀 ISRO Mission Q&A
- RAG (Retrieval-Augmented Generation) system using official ISRO documents
- Multi-query retrieval for comprehensive answers
- ChromaDB vector database for efficient document search
- Specialized knowledge about ISRO missions, satellites, and launch vehicles

### 🌍 Environmental Analysis
- Weather reports and summaries using NASA POWER API
- Flood risk assessment based on precipitation data
- Historical and current environmental data
- Support for specific dates, monthly periods, and 30-day analyses

## 🏗️ Architecture

### Core Components
1. **Intent Router (CrewAI-based)** – Autonomous agents analyze user requests and route them using Groq LLaMA models (with Groq-only fallback)
2. **Satellite Tracker** – Real-time orbital mechanics using Skyfield
3. **Document Q&A** – Vector-based retrieval with HuggingFace embeddings
4. **Environmental API** – NASA POWER data integration with analysis tools


### Tech Stack
- **Frontend**: Streamlit
- **AI/ML**: Groq API, LangChain, HuggingFace Embeddings
- **Vector DB**: ChromaDB
- **Satellite Tracking**: Skyfield, CelesTrak API
- **Environmental Data**: NASA POWER API
- **Visualization**: Plotly
- **Data Processing**: Pandas
- - **AI/Agents**: CrewAI + Groq (Llama3)
- **LLM Fallback**: Groq-only mode


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GROQ API key
- NASA API key (optional, defaults to DEMO_KEY)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd isro-intelligence-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
NASA_API_KEY=your_nasa_api_key_here
```

4. **Prepare the document database**
Ensure the `chroma_db` directory exists with your ISRO document embeddings.

5. **Run the application**
```bash
# Main unified assistant
streamlit run main_app.py

# Or Q&A only version
streamlit run qna_app.py
```

## 📋 Requirements

```txt
streamlit
requests
python-dotenv
groq
crewai
langchain-huggingface
langchain-community
langchain-groq
langchain-core
pandas
plotly
skyfield
chromadb
sentence-transformers

```

## 🎯 Usage Examples

### Satellite Tracking
```
"Where is ASTROSAT?"
"Track NVS-01"
"Show me the position of CARTOSAT-3"
```

### ISRO Mission Q&A
```
"What is the payload of Chandrayaan-1?"
"Tell me about PSLV"
"What are the objectives of Aditya-L1?"
```

### Environmental Analysis
```
"Weather in Bengaluru on July 15, 2025"
"Flood risk in Chennai for the last 30 days"
"Summary of Delhi weather for June 2025"
```

## 🛰️ Supported Satellites

### Active ISRO Satellites (25 tracked)
- **Navigation**: IRNSS-1B, IRNSS-1F, IRNSS-1I, NVS-01, NVS-02
- **Earth Observation**: ASTROSAT, SCATSAT-1, CARTOSAT series, RESOURCESAT-1, OCEANSAT series
- **Communication**: GSAT series, INSAT series
- **Scientific**: ADITYA-L1, CHANDRAYAAN-3 PROP

### Defunct Missions (Status Provided)
- Chandrayaan-1: Mission concluded (2009)
- Chandrayaan-2: Orbiter operational, lander/rover failed
- Mangalyaan (MOM): Contact lost (2022)

## 🗃️ Data Sources

### Primary Data Sources
- **ISRO Official Website**: Mission documentation and specifications
- **MOSDAC**: Earth observation and satellite data
- **CelesTrak**: Real-time satellite orbital elements
- **NASA POWER**: Environmental and agricultural data

### Document Processing
- Official ISRO mission reports and documentation
- Technical specifications and payload details
- Mission timelines and objectives
- Launch vehicle information

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Required for AI-powered intent routing and Q&A
- `NASA_API_KEY`: Optional for environmental data (defaults to DEMO_KEY)

### Caching Strategy
- Intent routing: 10-minute TTL
- Satellite TLE data: 5-minute TTL
- NASA environmental data: 1-hour TTL
- Vector database: Resource-level caching

### API Limitations
- NASA POWER API: 4-day data latency
- CelesTrak: Real-time TLE updates
- Groq API: Rate limits apply

## 🔍 Technical Details

### Intent Classification
The system uses a master router that classifies user queries into four categories:
1. `isro_qna`: Questions about ISRO missions and technology
2. `satellite_tracking`: Real-time satellite position requests
3. `agri_analysis`: Environmental and weather data analysis
4. `unknown`: Fallback for unrecognized queries

### Vector Database Setup
The Q&A system requires a pre-built ChromaDB with ISRO document embeddings:
- Embedding model: `BAAI/bge-large-en-v1.5`
- Vector store: ChromaDB with persistence
- Retrieval: Multi-query retriever with k=5

### Satellite Tracking Algorithm
1. Query normalization and fuzzy matching
2. NORAD ID lookup from curated satellite database
3. TLE data fetching from CelesTrak
4. Real-time position calculation using Skyfield
5. Coordinate conversion to latitude/longitude/altitude

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ISRO**: For providing comprehensive mission documentation
- **MOSDAC**: For satellite and earth observation data
- **NASA**: For environmental data through the POWER API
- **CelesTrak**: For real-time satellite tracking data
- **Groq**: For high-performance AI inference
- **HuggingFace**: For state-of-the-art embedding models

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check existing documentation and examples
- Review the troubleshooting section below

## 🐛 Troubleshooting

### Common Issues

**Q&A Database Not Found**
```
Error: ChromaDB directory 'chroma_db' not found
```
Solution: Ensure you have processed and embedded ISRO documents into ChromaDB

**API Key Issues**
```
Error: GROQ_API_KEY not found
```
Solution: Check your `.env` file and ensure proper API key configuration

**Satellite Not Found**
```
Error: Could not identify satellite
```
Solution: Check satellite name spelling or use fuzzy matching suggestions

**Environmental Data Unavailable**
```
Warning: Data not yet available from NASA
```
Solution: NASA POWER API has a 4-day latency; try an earlier date

## 🚀 Roadmap

- [ ] Additional satellite tracking sources
- [ ] Enhanced environmental analysis features
- [ ] Real-time mission status updates
- [ ] Mobile-responsive interface improvements
- [ ] Multi-language support
- [ ] API endpoint development
- [ ] Advanced visualization features

---

**Built with ❤️ for the Indian space community**
