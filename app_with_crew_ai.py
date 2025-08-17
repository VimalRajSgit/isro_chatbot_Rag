# ██████████████████████████████ 1. IMPORTS ██████████████████████████████
import os
import streamlit as st
import requests
import json
import time
import calendar
import difflib
from datetime import datetime, timedelta

# Environment & AI/ML
from dotenv import load_dotenv
import groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# CrewAI for intent recognition
# CrewAI for intent recognition
try:
    from crewai import Agent, Crew, Task
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_AVAILABLE = False
    st.warning(f"CrewAI import failed: {e}. Using Groq directly for intent recognition.")


# Data & Visualization
import pandas as pd
import plotly.graph_objects as go

# Satellite Tracking
from skyfield.api import load, EarthSatellite

# ██████████████████████████████ 2. INITIAL CONFIG & CONSTANTS ██████████████████████████████

# --- Load Environment Variables & Initialize Clients ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
groq_client = groq.Groq(api_key=GROQ_API_KEY)

# --- Constants ---
CONFIRMED_ACTIVE_ISRO = {
    39635: "IRNSS-1B", 41384: "IRNSS-1F", 43286: "IRNSS-1I", 56759: "NVS-01", 58831: "NVS-02",
    40930: "ASTROSAT", 42767: "SCATSAT-1", 43013: "CARTOSAT-2S", 44804: "CARTOSAT-2F",
    25994: "RESOURCESAT-1", 32060: "CARTOSAT-1", 49328: "CARTOSAT-3", 55086: "OCEANSAT-3",
    57320: "EOS-06", 35931: "OCEANSAT-2", 42698: "GSAT-9", 43241: "GSAT-6A", 44471: "GSAT-29",
    44506: "GSAT-11", 46278: "GSAT-31", 47438: "GSAT-12R", 27714: "INSAT-3A", 40730: "INSAT-3D",
    41469: "INSAT-3DR", 59051: "CHANDRAYAAN-3 PROP", 54684: "ADITYA-L1"
}
DEFUNCT_SATELLITES = {
    "chandrayaan-1": "Mission concluded in August 2009.",
    "chandrayaan-2 orbiter": "The orbiter is operational, but the lander and rover failed. Please query for mission details instead of tracking.",
    "mangalyaan": "Mission concluded after contact was lost in 2022.",
    "mom": "Mission concluded after contact was lost in 2022 (Mars Orbiter Mission)."
}
SATELLITE_NAME_LOOKUP = {name.lower().replace("-", "").replace("/", "").replace(" ", ""): (norad_id, name)
                         for norad_id, name in CONFIRMED_ACTIVE_ISRO.items()}
NASA_API_LATENCY_DAYS = 4


# ██████████████████████████████ 3. CORE LOGIC & HANDLERS ██████████████████████████████

# ========= CREWAI INTENT ROUTER =========
def initialize_intent_crew():
    """Initialize CrewAI crew for intent recognition"""

    if not CREWAI_AVAILABLE:
        return None

    try:
        # Create the intent recognition agent
        intent_agent = Agent(
            role="Intent Recognition Specialist",
            goal="Accurately identify user intent and extract relevant parameters from queries",
            backstory=f"""
            You are an expert at analyzing user queries and routing them to the appropriate service.
            Today's date is {datetime.now().strftime('%Y-%m-%d')}.

            You can identify these intents:
            1. isro_qna: Questions about ISRO missions, technologies, or history
            2. satellite_tracking: Requests for real-time satellite location/tracking
            3. agri_analysis: Requests for agricultural/environmental data analysis
            4. unknown: Queries that don't fit the above categories
            """,
            verbose=False,
            allow_delegation=False,
            llm=ChatGroq(temperature=0, model_name="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        )

        # Create the intent recognition task
        intent_task = Task(
            description="""
            Analyze the user query and determine the appropriate intent and parameters.

            Available intents and their parameters:

            1. `isro_qna`: For questions about ISRO missions, technologies, or history.
               - Example: "payload of chandrayan-1", "what is a pslv"
               - Required parameters: {"question": "The full user query"}

            2. `satellite_tracking`: For requests for real-time location or tracking of a satellite.
               - Example: "where is astrosat", "track nvs-01"
               - Required parameters: {"satellite_name": "The name of the satellite"}

            3. `agri_analysis`: For requests for agricultural or environmental data.
               - Keywords: "weather", "flood risk", "summary of", "drought", "soil moisture"
               - Required parameters: {"analysis_type": "type_of_analysis", "location": "The location specified"}
               - Optional parameters: {"date_period": "YYYY-MM-DD" or "YYYY-MM"}
               - analysis_type can be 'weather_report', 'weather_summary', or 'flood_risk'
               - Examples:
                 * "weather in bengaluru on july 15 2025" -> {"intent": "agri_analysis", "params": {"analysis_type": "weather_report", "location": "bengaluru", "date_period": "2025-07-15"}}
                 * "summary of bengaluru weather of whole june month 2025" -> {"intent": "agri_analysis", "params": {"analysis_type": "weather_summary", "location": "bengaluru", "date_period": "2025-06"}}
                 * "flood risk in bengaluru in june" -> {"intent": "agri_analysis", "params": {"analysis_type": "flood_risk", "location": "bengaluru", "date_period": "2025-06"}}

            4. `unknown`: If the query does not fit any of the above categories.
               - Required parameters: {}

            Query to analyze: {query}

            Respond ONLY with a valid JSON object in this format:
            {
                "intent": "intent_name",
                "params": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
            """,
            agent=intent_agent,
            expected_output="A valid JSON object with intent and parameters"
        )

        # Create and return the crew
        crew = Crew(
            agents=[intent_agent],
            tasks=[intent_task],
            verbose=False
        )

        return crew

    except Exception as e:
        st.error(f"Failed to initialize CrewAI: {str(e)}")
        return None


# Fallback function using Groq directly (original implementation)
def get_intent_with_groq(user_query: str) -> dict:
    """Fallback intent recognition using Groq directly"""
    system_prompt = f"""
    You are a master request router. Your job is to analyze the user's query and determine which tool to use.
    Today's date is {datetime.now().strftime('%Y-%m-%d')}.

    Here are the available intents and parameters:

    1.  `isro_qna`: For questions about ISRO missions, technologies, or history.
        - Example: "payload of chandrayan-1", "what is a pslv".
        - Required parameters: `{{ "question": "The full user query" }}`

    2.  `satellite_tracking`: For requests for the real-time location or tracking of a satellite.
        - Example: "where is astrosat", "track nvs-01".
        - Required parameters: `{{ "satellite_name": "The name of the satellite" }}`

    3.  `agri_analysis`: For requests for agricultural or environmental data.
        - Keywords: "weather", "flood risk", "summary of", "drought", "soil moisture".
        - Required parameters: `{{ "analysis_type": "type_of_analysis", "location": "The location specified" }}`
        - Optional parameters: `{{ "date_period": "YYYY-MM-DD" or "YYYY-MM" }}`. If the user asks for a specific date or a whole month, extract it.
        - `analysis_type` can be 'weather_report', 'weather_summary', or 'flood_risk'.
        - Example 1: "weather in bengaluru on july 15 2025" -> `{{ "intent": "agri_analysis", "params": {{"analysis_type": "weather_report", "location": "bengaluru", "date_period": "2025-07-15"}} }}`
        - Example 2: "summary of bengaluru weather of whole june month 2025" -> `{{ "intent": "agri_analysis", "params": {{"analysis_type": "weather_summary", "location": "bengaluru", "date_period": "2025-06"}} }}`
        - Example 3: "flood risk in bengaluru in june" -> `{{ "intent": "agri_analysis", "params": {{"analysis_type": "flood_risk", "location": "bengaluru", "date_period": "2025-06"}} }}`

    4.  `unknown`: If the query does not fit any of the above categories.
        - Required parameters: `{{}}`

    Respond ONLY with a valid JSON object.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            model="llama3-70b-8192", temperature=0.0, response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        return {"intent": "unknown", "params": {}}


def get_intent(user_query: str) -> dict:
    """Get intent using CrewAI crew or fallback to Groq"""
    if not CREWAI_AVAILABLE:
        # Use direct Groq implementation as fallback
        return get_intent_with_groq(user_query)

    try:
        crew = initialize_intent_crew()
        if crew is None:
            # Fallback to direct Groq if CrewAI fails
            return get_intent_with_groq(user_query)

        result = crew.kickoff(inputs={"query": user_query})

        # Parse the result - CrewAI returns a CrewOutput object
        # We need to extract the raw string and parse it as JSON
        result_str = str(result.raw) if hasattr(result, 'raw') else str(result)

        # Clean the result string and try to find JSON
        import re
        # Remove any markdown formatting
        result_str = re.sub(r'```json\n?', '', result_str)
        result_str = re.sub(r'```\n?', '', result_str)
        result_str = result_str.strip()

        # Try to find JSON in the result string
        json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if json_match:
            json_str = json_match.group().strip()
            return json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire cleaned result
            return json.loads(result_str)

    except Exception as e:
        st.warning(f"CrewAI intent recognition failed: {str(e)}. Using fallback.")
        # Fallback to direct Groq implementation
        return get_intent_with_groq(user_query)


# ========= TOOL-SPECIFIC HELPER FUNCTIONS =========
@st.cache_resource
def initialize_qna_chain():
    # ... (same as before)
    if not os.path.exists("chroma_db"): return None
    embedding_fn = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'},
                                         encode_kwargs={'normalize_embeddings': True})
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_fn)
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)
    retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), llm=llm)
    prompt_template = "You are an assistant for ISRO question-answering. Use the retrieved context to answer the question. If you don't know the answer, say that. Keep the answer concise and under 4 sentences.\nCONTEXT: {context}\nQUESTION: {question}\nANSWER:"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()


@st.cache_data(ttl=300)
def get_tle_data(norad_id):
    # ... (same as before)
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
    try:
        r = requests.get(url, timeout=5);
        r.raise_for_status()
        if len(r.text) > 100:
            lines = r.text.strip().split('\n');
            return lines[0].strip(), lines[1].strip(), lines[2].strip()
    except requests.RequestException:
        return None, None, None
    return None, None, None


@st.cache_data(ttl=3600)
def fetch_nasa_data(lat, lon, start, end):
    # ... (same as before)
    params = {"parameters": "T2M,RH2M,PRECTOTCORR,T2M_MAX,T2M_MIN", "community": "AG", "format": "JSON",
              "latitude": lat, "longitude": lon, "start": start, "end": end, "api_key": NASA_API_KEY}
    try:
        r = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params);
        r.raise_for_status()
        data = r.json()
        if not data.get('properties', {}).get('parameter'): return None
        params_data = data['properties']['parameter']
        timestamps = list(next(iter(params_data.values())).keys())
        return pd.DataFrame(
            [{'date': pd.to_datetime(ts, format='%Y%m%d'), **{p: vals.get(ts) for p, vals in params_data.items()}} for
             ts in timestamps])
    except Exception:
        return None


# ========= HANDLER FUNCTIONS (PERFORM ACTIONS & DISPLAY OUTPUT) =========

def handle_qna(params):
    # ... (same as before)
    question = params.get("question")
    rag_chain = initialize_qna_chain()
    if not rag_chain: st.error("Q&A database (`chroma_db`) not found."); return
    with st.spinner("Searching ISRO documents..."):
        response = rag_chain.invoke(question)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def handle_satellite_tracking(params):
    satellite_name = params.get("satellite_name", "").lower().replace(" ", "")

    # **NEW**: Check for defunct satellites first
    if satellite_name in DEFUNCT_SATELLITES:
        response = f"**{satellite_name.title()}** cannot be tracked. {DEFUNCT_SATELLITES[satellite_name]}"
        st.warning(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return

    matches = difflib.get_close_matches(satellite_name, SATELLITE_NAME_LOOKUP.keys(), n=1, cutoff=0.7)
    if not matches:
        st.error(f"Sorry, I could not identify '{params.get('satellite_name')}' in my active list.")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Could not identify satellite '{params.get('satellite_name')}'."})
        return

    norad_id, official_name = SATELLITE_NAME_LOOKUP[matches[0]]
    with st.spinner(f"Acquiring live telemetry for **{official_name}**..."):
        name_line, l1, l2 = get_tle_data(norad_id)
        if not l1 or not l2:
            st.error(f"Could not fetch live tracking data for {official_name}.");
            return

        ts = load.timescale();
        satellite = EarthSatellite(l1, l2, name_line, ts);
        current_time = ts.now()
        geocentric = satellite.at(current_time);
        subpoint = geocentric.subpoint()

        st.subheader(f"🛰️ Live Position: {official_name}")
        col1, col2, col3 = st.columns(3);
        col1.metric("Latitude", f"{subpoint.latitude.degrees:.4f}°")
        col2.metric("Longitude", f"{subpoint.longitude.degrees:.4f}°");
        col3.metric("Altitude", f"{subpoint.elevation.km:.2f} km")
        st.caption(f"Data acquired at {current_time.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")

        response_text = f"Live position for **{official_name}**: Lat={subpoint.latitude.degrees:.2f}°, Lon={subpoint.longitude.degrees:.2f}°, Alt={subpoint.elevation.km:.0f} km."
        st.session_state.messages.append({"role": "assistant", "content": response_text})


def assess_flood_risk(df, location, period_str):
    """Simple flood risk assessment based on monthly precipitation."""
    total_precip = df['PRECTOTCORR'].sum()
    heavy_rain_days = df[df['PRECTOTCORR'] > 25].shape[0]  # Days with > 25mm rain

    risk = "Low"
    if total_precip > 400 or heavy_rain_days > 5:
        risk = "High"
    elif total_precip > 250 or heavy_rain_days > 2:
        risk = "Moderate"

    st.subheader(f"🌊 Flood Risk for {location.title()} in {period_str}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Assessment", risk)
    col2.metric("Total Precipitation", f"{total_precip:.1f} mm")
    col3.metric("Heavy Rain Days (>25mm)", f"{heavy_rain_days}")

    response_text = f"The estimated flood risk for **{location.title()}** in **{period_str}** is **{risk}**, with a total precipitation of {total_precip:.1f} mm."
    return response_text


def handle_agri_analysis(params):
    # **REFACTORED** to handle months and specific analysis types
    location = params.get("location")
    date_period = params.get("date_period")
    analysis_type = params.get("analysis_type", "weather_report")
    most_recent_available_date = datetime.now() - timedelta(days=NASA_API_LATENCY_DAYS)

    if not location: st.error("A location is required for environmental analysis."); return

    with st.spinner(f"Performing {analysis_type.replace('_', ' ')} for {location}..."):
        try:
            coord_completion = groq_client.chat.completions.create(messages=[{"role": "system",
                                                                              "content": "You are a geocoding expert. Given a location, respond ONLY with a JSON object like {\"lat\": <latitude>, \"lon\": <longitude>}."},
                                                                             {"role": "user", "content": location}],
                                                                   model="llama3-8b-8192", temperature=0.0,
                                                                   response_format={"type": "json_object"})
            coords = json.loads(coord_completion.choices[0].message.content)
        except Exception:
            st.error(f"Could not find coordinates for '{location}'.");
            return

        start_date, end_date = None, None
        period_str = ""
        is_monthly = False

        if date_period and len(date_period) == 7:  # Monthly: "YYYY-MM"
            is_monthly = True
            year, month = map(int, date_period.split('-'))
            start_date = datetime(year, month, 1)
            _, num_days = calendar.monthrange(year, month)
            end_date = datetime(year, month, num_days)
            period_str = start_date.strftime('%B %Y')
        elif date_period:  # Daily: "YYYY-MM-DD"
            start_date = end_date = datetime.strptime(date_period, "%Y-%m-%d")
            period_str = start_date.strftime('%B %d, %Y')

        if end_date and end_date > most_recent_available_date:
            st.warning(f"Data for the requested period ({period_str}) is not yet available from NASA.");
            return

        if not start_date:  # Default to last 30 available days
            end_date = most_recent_available_date
            start_date = end_date - timedelta(days=29)
            period_str = "Last 30 Available Days"

        df = fetch_nasa_data(coords['lat'], coords['lon'], start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        if df is None or df.empty: st.error(f"Could not retrieve data from NASA for '{location}'."); return

        if is_monthly and analysis_type == 'flood_risk':
            response_text = assess_flood_risk(df, location, period_str)
        elif is_monthly and analysis_type == 'weather_summary':
            st.subheader(f"🌦️ Monthly Summary for {location.title()} in {period_str}")
            avg_temp = df['T2M'].mean();
            total_precip = df['PRECTOTCORR'].sum()
            col1, col2 = st.columns(2);
            col1.metric("Avg Monthly Temp", f"{avg_temp:.1f}°C");
            col2.metric("Total Monthly Precip", f"{total_precip:.1f} mm")
            response_text = f"In **{period_str}**, **{location.title()}** had an average temperature of {avg_temp:.1f}°C and total precipitation of {total_precip:.1f} mm."
        else:  # Default to single day report or 30 day summary if not monthly
            st.subheader(f"🌤️ Weather for {location.title()} on {period_str}")
            data = df.iloc[0] if len(df) == 1 else df
            avg_temp = data['T2M'].mean();
            total_precip = data['PRECTOTCORR'].sum()
            col1, col2 = st.columns(2);
            col1.metric("Avg Temp", f"{avg_temp:.1f}°C");
            col2.metric("Total Precip", f"{total_precip:.1f} mm")
            response_text = f"Weather for **{location.title()}** for the period **{period_str}**: Avg Temp was {avg_temp:.1f}°C, Total Precip was {total_precip:.1f} mm."

        st.session_state.messages.append({"role": "assistant", "content": response_text})


def handle_unknown():
    # ... (same as before)
    response = "I can help with: answering questions about ISRO, providing real-time satellite tracking, or performing environmental analysis. Please try one of those."
    st.markdown(response);
    st.session_state.messages.append({"role": "assistant", "content": response})


# ██████████████████████████████ 4. STREAMLIT APP UI ██████████████████████████████
def main():
    st.set_page_config(page_title="Unified Intelligence Assistant", page_icon="🌌", layout="centered")
    st.title("🌌 Unified Intelligence Assistant")
    st.markdown("Ask about ISRO missions, track satellites, or request environmental analysis—all in one place.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing request..."):
                if not GROQ_API_KEY or not NASA_API_KEY: st.error(
                    "API Key is missing. Check your .env file."); st.stop()

                intent_data = get_intent(prompt)
                intent = intent_data.get("intent");
                params = intent_data.get("params", {})

                if intent == "isro_qna":
                    handle_qna(params)
                elif intent == "satellite_tracking":
                    handle_satellite_tracking(params)
                elif intent == "agri_analysis":
                    handle_agri_analysis(params)
                else:
                    handle_unknown()


if __name__ == "__main__":
    main()
