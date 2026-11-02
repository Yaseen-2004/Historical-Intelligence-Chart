
import os
import re
import time
import pickle
import streamlit as st
import numpy as np

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HistoryGPT | Historical Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATHS = [
    BASE_DIR / "models" / "retrieval_index.pkl",
    BASE_DIR / "retrieval_index.pkl"
]

INDEX_PATH = next(
    (p for p in INDEX_PATHS if p.exists()),
    None
)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ==========================================
# CUSTOM 3D ANIMATED DESIGN SYSTEM
# ==========================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap'
);

:root {
    --background: #070914;
    --surface: #101426;
    --purple: #9B7BFF;
    --cyan: #49D8FF;
    --text: #F2F3FF;
    --muted: #969CB8;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(93, 62, 180, 0.20),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 30%,
            rgba(34, 159, 214, 0.12),
            transparent 30%
        ),
        #070914 !important;

    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101426,
        #090C19
    );
    border-right: 1px solid rgba(155,123,255,0.17);
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

p, label, li {
    color: #C5C9DC;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Animated background orbs */

.orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    filter: blur(3px);
    z-index: 0;
    animation: floatOrb 9s ease-in-out infinite;
}

.orb-one {
    width: 190px;
    height: 190px;
    background: radial-gradient(
        circle,
        rgba(139, 92, 246, 0.15),
        transparent 70%
    );
    top: 12%;
    right: 8%;
}

.orb-two {
    width: 250px;
    height: 250px;
    background: radial-gradient(
        circle,
        rgba(34, 211, 238, 0.11),
        transparent 70%
    );
    bottom: 8%;
    left: 15%;
    animation-delay: 3s;
}

@keyframes floatOrb {
    0%, 100% {
        transform: translate3d(0, 0, 0);
    }
    50% {
        transform: translate3d(25px, -30px, 40px);
    }
}

/* Main hero */

.hero {
    position: relative;
    padding: 38px;
    border-radius: 25px;
    overflow: hidden;

    background: linear-gradient(
        125deg,
        rgba(28, 27, 61, 0.96),
        rgba(15, 25, 48, 0.95),
        rgba(16, 17, 40, 0.96)
    );

    border: 1px solid rgba(160, 137, 255, 0.30);

    box-shadow:
        0 20px 65px rgba(0,0,0,0.35),
        inset 0 1px 1px rgba(255,255,255,0.10);

    transform-style: preserve-3d;
    animation: heroEntrance 1s ease-out;
}

.hero::before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    right: -90px;
    top: -180px;
    border-radius: 50%;

    background: radial-gradient(
        circle,
        rgba(155,123,255,0.38),
        transparent 70%
    );

    animation: pulse 5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
        opacity: 0.5;
    }
    50% {
        transform: scale(1.3);
        opacity: 1;
    }
}

@keyframes heroEntrance {
    from {
        opacity: 0;
        transform: perspective(1000px) rotateX(8deg) translateY(30px);
    }
    to {
        opacity: 1;
        transform: perspective(1000px) rotateX(0deg) translateY(0);
    }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 5vw, 55px);
    font-weight: 800;
    letter-spacing: -2px;

    background: linear-gradient(
        90deg,
        #FFFFFF,
        #B8A4FF,
        #64DDFF
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #ADB6D5;
    font-size: 15px;
    max-width: 650px;
    line-height: 1.8;
}

.hero-tag {
    display: inline-block;
    padding: 7px 13px;
    margin-bottom: 15px;
    border-radius: 30px;

    color: #CBBEFF;
    background: rgba(155,123,255,0.13);
    border: 1px solid rgba(155,123,255,0.30);

    font-size: 11px;
    letter-spacing: 1.5px;
}

/* 3D cards */

.metric-card {
    padding: 22px;
    min-height: 125px;
    border-radius: 18px;

    background: linear-gradient(
        145deg,
        rgba(31,36,65,0.90),
        rgba(14,19,37,0.94)
    );

    border: 1px solid rgba(160,160,255,0.17);

    box-shadow:
        0 12px 25px rgba(0,0,0,0.22),
        inset 0 1px 0 rgba(255,255,255,0.08);

    transition: all 0.35s ease;
}

.metric-card:hover {
    transform:
        perspective(900px)
        rotateX(5deg)
        rotateY(-5deg)
        translateY(-8px);

    border-color: rgba(155,123,255,0.60);

    box-shadow:
        0 25px 45px rgba(100,80,220,0.20);
}

.metric-label {
    color: #9CA5C2;
    font-size: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 29px;
    font-weight: 700;
    margin-top: 12px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 12px;

    border: 1px solid rgba(155,123,255,0.5);

    background: linear-gradient(
        110deg,
        #7754D8,
        #4B91D9
    );

    color: white;
    font-weight: 600;

    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(116,84,220,0.35);
    border-color: #C3B2FF;
}

/* Inputs */

.stTextInput input,
.stTextArea textarea {
    background: rgba(17,22,43,0.9) !important;
    border: 1px solid rgba(155,123,255,0.25) !important;
    color: white !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessage"] {
    background: rgba(22,27,50,0.72);
    border: 1px solid rgba(155,123,255,0.13);
    border-radius: 17px;
    padding: 12px;
    margin-bottom: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #B7BBD0;
}

.stTabs [aria-selected="true"] {
    color: #BBAAFF !important;
}

hr {
    border-color: rgba(155,123,255,0.15);
}


/* HISTORYGPT HERO ENHANCEMENTS */

.hero {
    position: relative;
    padding: 45px 10px 35px;
    perspective: 1000px;
}

.hero-tag {
    display: inline-block;
    padding: 10px 18px;
    margin-bottom: 24px;
    border: 1px solid rgba(0, 229, 255, 0.4);
    border-radius: 30px;
    background: rgba(0, 229, 255, 0.08);
    color: #00e5ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    box-shadow: 0 0 22px rgba(0, 229, 255, 0.12);
    animation: heroFade 1s ease-out;
}

.hero-title {
    font-size: clamp(38px, 6vw, 68px);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -2px;
    margin-bottom: 22px;
    background: linear-gradient(
        110deg,
        #ffffff 10%,
        #64e8ff 45%,
        #a78bfa 75%,
        #ffffff 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: heroGradient 5s linear infinite;
}

.hero-subtitle {
    max-width: 760px;
    color: #a6b3c9;
    font-size: 16px;
    line-height: 1.9;
    letter-spacing: 0.25px;
    animation: heroFade 1.4s ease-out;
}

@keyframes heroGradient {
    to {
        background-position: 200% center;
    }
}

@keyframes heroFade {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>

<div class="orb orb-one"></div>
<div class="orb orb-two"></div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD KNOWLEDGE BASE
# ==========================================

@st.cache_resource
def load_knowledge_base():

    if INDEX_PATH is None:
        raise FileNotFoundError(
            "Retrieval index not found. "
            "Please save retrieval_index.pkl in the models folder."
        )

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    model_name = data.get(
        "embedding_model_name",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_model = SentenceTransformer(model_name)

    return data, embedding_model


try:
    retrieval_data, embedding_model = load_knowledge_base()

    historical_chunks = retrieval_data["historical_chunks"]
    chunk_embeddings = retrieval_data["chunk_embeddings"]
    tfidf_vectorizer = retrieval_data["tfidf_vectorizer"]
    tfidf_matrix = retrieval_data["tfidf_matrix"]

    knowledge_ready = True
    knowledge_error = None

except Exception as e:
    knowledge_ready = False
    knowledge_error = str(e)


# ==========================================
# HYBRID RETRIEVAL
# ==========================================

def retrieve_context(question, top_k=5):

    query_tfidf = tfidf_vectorizer.transform([question])

    keyword_scores = cosine_similarity(
        query_tfidf,
        tfidf_matrix
    )[0]

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    scores = (
        0.35 * keyword_scores +
        0.65 * semantic_scores
    )

    ranked = np.argsort(scores)[::-1]

    selected = []
    selected_embeddings = []

    for index in ranked:

        candidate = chunk_embeddings[index]

        if selected_embeddings:
            similarities = cosine_similarity(
                [candidate],
                selected_embeddings
            )[0]

            if max(similarities) >= 0.90:
                continue

        selected.append(index)
        selected_embeddings.append(candidate)

        if len(selected) >= top_k:
            break

    results = []

    for index in selected:
        results.append({
            "title": historical_chunks[index]["title"],
            "url": historical_chunks[index]["url"],
            "text": historical_chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# ==========================================
# GEMINI GENERATION
# ==========================================

def get_api_key():

    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def generate_answer(question, context):

    api_key = get_api_key()

    if not api_key:
        return None, "API_KEY_MISSING"

    reference_text = ""

    for i, item in enumerate(context, 1):
        reference_text += f"""
REFERENCE {i}
Title: {item['title']}
Source: {item['url']}
Evidence:
{item['text'][:1800]}
"""

    prompt = f"""
You are HistoryGPT, an educational historical research assistant.

Answer the question using the supplied historical references.

Rules:
- Never invent facts or dates.
- Do not introduce unsupported claims.
- Clearly mention uncertainty where appropriate.
- Explain in student-friendly language.
- Use headings and bullet points when useful.
- Include source titles and URLs.
- If evidence is insufficient, say so.

HISTORICAL EVIDENCE:
{reference_text}

QUESTION:
{question}

Write the answer.
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1200
            )
        )

        if not response.text:
            return None, "EMPTY_RESPONSE"

        return response.text, "SUCCESS"

    except Exception as e:

        message = str(e).lower()

        if "429" in message or "resource_exhausted" in message:
            return None, "QUOTA_EXHAUSTED"

        return None, f"GENERATION_ERROR: {str(e)[:250]}"


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "questions_count" not in st.session_state:
    st.session_state.questions_count = 0


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <div style="
            font-size:27px;
            font-weight:800;
            font-family:'Space Grotesk';
            background:linear-gradient(90deg,#B9A5FF,#5BDFFF);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;">
            ◈ HISTORYGPT
        </div>
        <div style="color:#929BB9;font-size:11px;letter-spacing:2px;">
            HISTORICAL INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ◈ Navigation")

    page = st.radio(
        "Workspace",
        [
            "AI Chat",
            "Time Explorer",
            "Knowledge Arena",
            "Flashcard Studio",
            "Research Notebook",
            "Analytics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### SYSTEM STATUS")

    if knowledge_ready:
        st.success("Knowledge base online")
    else:
        st.error("Knowledge base unavailable")

    if get_api_key():
        st.success("Gemini key configured")
    else:
        st.warning("Retrieval-only mode")

    st.markdown("---")

    if st.button("＋ New Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption("HistoryGPT • Educational AI Platform")
    st.caption("Powered by Hybrid RAG + Gemini")


# ==========================================
# HERO SECTION
# ==========================================

st.html("""
<div class="hero">

    <div class="hero-tag">
        ✦ AI-POWERED HISTORICAL INTELLIGENCE
    </div>

    <div class="hero-title">
        Explore the Past.<br>
        Understand History.
    </div>

    <div class="hero-subtitle">
        Discover civilizations, historical personalities, wars,
        discoveries and forgotten stories through an evidence-grounded
        artificial intelligence experience.
    </div>

</div>
""")

st.write("")


# ==========================================
# METRIC CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Knowledge Articles</div>
        <div class="metric-value">
            {len(set(x['title'] for x in historical_chunks))
             if knowledge_ready else 0}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical Passages</div>
        <div class="metric-value">
            {len(historical_chunks) if knowledge_ready else 0}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Questions This Session</div>
        <div class="metric-value">
            {st.session_state.questions_count}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ==========================================
# KNOWLEDGE BASE ERROR
# ==========================================

if not knowledge_ready:
    st.error(f"Knowledge base error: {knowledge_error}")


# ==========================================
# AI CHAT
# ==========================================

if page == "AI Chat":

    st.markdown("## ◈ Historical Intelligence Chat")
    st.caption(
        "Ask questions about civilizations, empires, rulers and events."
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("View historical evidence and sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**{source['title']}** "
                            f"— Relevance: {source['score']:.3f}"
                        )

                        st.write(source["text"][:900])

                        st.markdown(
                            f"[Open Wikipedia source]({source['url']})"
                        )

                        st.divider()

    question = st.chat_input(
        "Ask HistoryGPT anything about history..."
    )

    if question:

        if not knowledge_ready:
            st.error("Historical knowledge base is not available.")

        else:

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            st.session_state.questions_count += 1

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching historical archives..."
                ):

                    context = retrieve_context(question, top_k=5)

                if not context:

                    answer = (
                        "I could not find sufficiently relevant "
                        "historical passages in the current knowledge base."
                    )

                else:

                    with st.spinner(
                        "Analyzing historical evidence..."
                    ):

                        answer, status = generate_answer(
                            question,
                            context
                        )

                    if status == "SUCCESS":

                        pass

                    elif status == "API_KEY_MISSING":

                        answer = (
                            "The historical retrieval engine successfully "
                            "found relevant evidence, but Gemini is not "
                            "configured. Configure GEMINI_API_KEY to enable "
                            "AI-generated answers. You can inspect the "
                            "retrieved evidence below."
                        )

                    elif status == "QUOTA_EXHAUSTED":

                        answer = (
                            "Gemini's API quota is currently exhausted. "
                            "The retrieval system is working, and the "
                            "historical evidence is available below. "
                            "Try generation again after your quota resets "
                            "or after enabling an eligible paid tier."
                        )

                    else:

                        answer = (
                            f"Answer generation unavailable: {status}. "
                            "Retrieved historical evidence is shown below."
                        )

                st.markdown(answer)

                if context:

                    with st.expander(
                        "Explore retrieved historical evidence"
                    ):

                        for source in context:

                            st.markdown(
                                f"**{source['title']}** "
                                f"({source['score']:.3f})"
                            )

                            st.write(source["text"][:1000])

                            st.markdown(
                                f"[Wikipedia article]({source['url']})"
                            )

                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": context
            })

            st.rerun()


# ==========================================
# TIME EXPLORER
# ==========================================

elif page == "Time Explorer":

    st.markdown("## ◈ Time Explorer")
    st.caption("Explore important historical eras.")

    historical_eras = {
        "Ancient Civilizations": [
            ("Ancient Egypt", "c. 3100 BCE"),
            ("Indus Valley Civilisation", "c. 3300–1300 BCE"),
            ("Maurya Empire", "c. 322–185 BCE"),
            ("Roman Empire", "27 BCE–476 CE")
        ],
        "Medieval History": [
            ("Gupta Empire", "c. 320–550 CE"),
            ("Mughal Empire", "1526–1857 CE")
        ],
        "Modern History": [
            ("French Revolution", "1789–1799"),
            ("Industrial Revolution", "18th–19th century"),
            ("World War I", "1914–1918"),
            ("World War II", "1939–1945")
        ]
    }

    selected_era = st.selectbox(
        "Choose a historical era",
        list(historical_eras.keys())
    )

    for title, date in historical_eras[selected_era]:

        st.markdown(f"""
        <div class="metric-card" style="margin:12px 0;">
            <div class="metric-label">{date}</div>
            <div class="metric-value" style="font-size:22px;">
                {title}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# KNOWLEDGE ARENA
# ==========================================

elif page == "Knowledge Arena":

    st.markdown("## ◈ Knowledge Arena")
    st.caption("Test your understanding of history.")

    quiz_questions = [
        {
            "question": "Who was the third ruler of the Maurya Empire?",
            "options": [
                "Chandragupta Maurya",
                "Ashoka",
                "Bindusara",
                "Brihadratha"
            ],
            "answer": "Ashoka"
        },
        {
            "question": "Which war influenced Ashoka's transformation?",
            "options": [
                "Battle of Plassey",
                "Kalinga War",
                "Battle of Panipat",
                "Battle of Hydaspes"
            ],
            "answer": "Kalinga War"
        },
        {
            "question": "Which event began in 1914?",
            "options": [
                "World War II",
                "French Revolution",
                "World War I",
                "Cold War"
            ],
            "answer": "World War I"
        }
    ]

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    for i, item in enumerate(quiz_questions):

        st.markdown(f"**Question {i+1}. {item['question']}**")

        selected = st.radio(
            f"Choose your answer {i+1}",
            item["options"],
            key=f"quiz_{i}"
        )

        st.session_state.quiz_answers[i] = selected

    if st.button("Submit Quiz"):

        score = sum(
            st.session_state.quiz_answers.get(i) == q["answer"]
            for i, q in enumerate(quiz_questions)
        )

        st.success(
            f"Your score: {score}/{len(quiz_questions)}"
        )


# ==========================================
# FLASHCARD STUDIO
# ==========================================

elif page == "Flashcard Studio":

    st.markdown("## ◈ Flashcard Studio")

    flashcards = [
        {
            "front": "Who was Ashoka?",
            "back": "The third ruler of the Maurya Empire."
        },
        {
            "front": "What was the Kalinga War?",
            "back": "A war fought by Ashoka against Kalinga."
        },
        {
            "front": "What was the Maurya Empire's capital?",
            "back": "Pataliputra."
        }
    ]

    if "flash_index" not in st.session_state:
        st.session_state.flash_index = 0

    if "flash_reveal" not in st.session_state:
        st.session_state.flash_reveal = False

    card = flashcards[st.session_state.flash_index]

    st.markdown(f"""
    <div class="hero" style="text-align:center;padding:65px 25px;">
        <div class="hero-tag">REVISION CARD</div>
        <div style="font-size:12px;color:#A9B0CD;">
            CARD {st.session_state.flash_index + 1}
            OF {len(flashcards)}
        </div>
        <div style="font-size:27px;font-weight:700;margin:30px 0;">
            {card['back'] if st.session_state.flash_reveal else card['front']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    a, b = st.columns(2)

    with a:
        if st.button("Reveal / Hide Answer"):
            st.session_state.flash_reveal = (
                not st.session_state.flash_reveal
            )
            st.rerun()

    with b:
        if st.button("Next Card"):
            st.session_state.flash_index = (
                st.session_state.flash_index + 1
            ) % len(flashcards)

            st.session_state.flash_reveal = False
            st.rerun()


# ==========================================
# RESEARCH NOTEBOOK
# ==========================================

elif page == "Research Notebook":

    st.markdown("## ◈ Research Notebook")

    if "notes" not in st.session_state:
        st.session_state.notes = ""

    st.session_state.notes = st.text_area(
        "Your historical research notes",
        value=st.session_state.notes,
        height=300,
        placeholder="Write your historical notes here..."
    )

    if st.button("Save Notes"):

        notes_path = BASE_DIR / "database" / "research_notes.txt"

        notes_path.write_text(
            st.session_state.notes,
            encoding="utf-8"
        )

        st.success("Research notes saved successfully.")


# ==========================================
# ANALYTICS
# ==========================================

elif page == "Analytics":

    st.markdown("## ◈ Learning Analytics")

    st.metric(
        "Questions asked this session",
        st.session_state.questions_count
    )

    st.metric(
        "Knowledge passages",
        len(historical_chunks) if knowledge_ready else 0
    )

    st.info(
        "Detailed learning analytics and persistent user progress "
        "will be added in the next development phase."
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    padding:18px;
    color:#78829F;
    font-size:12px;
">
    HISTORYGPT · EXPLORE THE PAST · POWERED BY ARTIFICIAL INTELLIGENCE
    <br><br>
    Historical knowledge references are sourced from Wikipedia.
    Wikipedia content is available under its applicable licensing terms.
</div>
""", unsafe_allow_html=True)
