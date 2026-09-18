"""
City Intelligence System — Streamlit UI
Wraps a LangChain + Groq tool-calling agent (weather + news) with a
premium chat interface and human-in-the-loop tool confirmation.

Run with:  streamlit run app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="City Intel",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Premium CSS — dark glassmorphism theme
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-0: #0a0e14;
        --bg-1: #10151d;
        --bg-card: rgba(255,255,255,0.035);
        --border: rgba(255,255,255,0.08);
        --accent: #5eead4;
        --accent-2: #818cf8;
        --text-hi: #f1f5f9;
        --text-lo: #94a3b8;
        --danger: #fb7185;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(94,234,212,0.08), transparent 40%),
            radial-gradient(circle at 85% 20%, rgba(129,140,248,0.08), transparent 40%),
            var(--bg-0);
        color: var(--text-hi);
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--border);
    }

    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
        border-radius: 18px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        backdrop-filter: blur(12px);
    }
    .app-header .badge {
        font-size: 28px;
        width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }
    .app-header h1 {
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        color: var(--text-hi);
    }
    .app-header p {
        margin: 2px 0 0 0;
        font-size: 13px;
        color: var(--text-lo);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 4px 6px !important;
        backdrop-filter: blur(10px);
    }

    /* Tool result cards */
    .tool-card {
        border-radius: 14px;
        padding: 16px 18px;
        margin: 8px 0;
        background: linear-gradient(135deg, rgba(94,234,212,0.06), rgba(129,140,248,0.04));
        border: 1px solid var(--border);
    }
    .tool-card .tool-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }
    .tool-card .tool-body {
        font-size: 14px;
        color: var(--text-hi);
        line-height: 1.5;
        white-space: pre-wrap;
    }

    .tool-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        margin: 4px 0 8px 4px;
    }

    /* Individual news article card */
    .news-card {
        border-radius: 12px;
        padding: 12px 16px 4px 16px;
        margin: 6px 0 2px 0;
        background: rgba(129,140,248,0.05);
        border: 1px solid var(--border);
    }
    .news-card .news-title {
        font-weight: 600;
        font-size: 14.5px;
        color: var(--text-hi);
        margin-bottom: 4px;
    }
    .news-card .news-snippet {
        font-size: 13px;
        color: var(--text-lo);
        line-height: 1.5;
    }
    .news-card .news-domain {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        color: var(--accent-2);
        margin: 6px 0 2px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pending confirmation card */
    .confirm-card {
        border-radius: 14px;
        padding: 14px 18px;
        margin: 10px 0;
        background: rgba(251,113,133,0.06);
        border: 1px solid rgba(251,113,133,0.3);
    }
    .confirm-card .confirm-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--danger);
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .confirm-card .confirm-args {
        font-size: 13px;
        color: var(--text-lo);
        margin-top: 4px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        color: var(--text-hi);
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Sidebar stat chips */
    .stat-chip {
        display: flex;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 10px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        font-size: 13px;
        margin-bottom: 8px;
    }
    .stat-chip span:first-child { color: var(--text-lo); }
    .stat-chip span:last-child { color: var(--accent); font-weight: 600; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Tools
# ────────────────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get current weather of the city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Couldn't fetch weather for {city}: {data.get('message', 'unknown error')}"

    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"The weather in {city}: {description}, {temp}°C"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Markers that signal "junk starts here" in Tavily's scraped content
# (nav menus, trending-topic rails, e-commerce widgets bleeding into the text)
_JUNK_MARKERS = [
    "latest news", "trending stories", "top trending", "trending",
    "home ›", "select variant", "choose your city", "write a review",
    "tired of too many ads",
]


def _clean_snippet(text: str, max_len: int = 220) -> str:
    """Strip common scraper junk and trim to a clean, short snippet."""
    if not text:
        return ""
    text = text.replace("[...]", " ").replace("\n", " ")
    text = " ".join(text.split())  # collapse whitespace

    lowered = text.lower()
    cut_at = len(text)
    for marker in _JUNK_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    text = text[:cut_at].strip(" -|")

    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip(",.") + "…"
    return text


def _domain(url: str) -> str:
    try:
        return url.split("//")[-1].split("/")[0].replace("www.", "")
    except Exception:
        return url


@tool
def get_news(city: str) -> str:
    """Get the latest news about a city"""
    query = f"Get the top 5 latest news about the {city}"
    response = tavily_client.search(
        query=query, search_depth="basic", max_results=5, topic="news"
    )
    data = response.get("results")

    if not data:
        st.session_state["_last_news_articles"] = []
        return f"No recent news found for {city}."

    articles = []
    for news in data:
        title = news.get("title", "").strip()
        url = news.get("url", "")
        snippet = _clean_snippet(news.get("content", ""))
        if not snippet:
            continue  # drop entries that are pure junk after cleaning
        articles.append({"title": title, "url": url, "snippet": snippet, "domain": _domain(url)})

    # Stash structured, clean data for the UI to render as cards.
    st.session_state["_last_news_articles"] = articles
    st.session_state["_last_news_city"] = city

    if not articles:
        return f"No usable news found for {city} after filtering out junk results."

    # Compact plain-text version is what actually goes to the model.
    compact = "\n".join(f"- {a['title']}: {a['snippet']}" for a in articles)
    return f"Latest news in {city}:\n{compact}"


TOOLS = {"get_news": get_news, "get_weather": get_weather}

# ────────────────────────────────────────────────────────────────────────────
# LLM
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return llm.bind_tools([get_news, get_weather])


llm_with_tools = get_llm()

# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # LangChain message objects (model context)
if "display_log" not in st.session_state:
    st.session_state.display_log = []       # what gets rendered: dicts {role, kind, content}
if "pending_tool_calls" not in st.session_state:
    st.session_state.pending_tool_calls = None   # list of tool_call dicts awaiting confirmation
if "stats" not in st.session_state:
    st.session_state.stats = {"queries": 0, "tools_run": 0, "tools_skipped": 0}

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Control Panel")
    st.caption("Weather + news agent · human-in-the-loop tool calls")

    st.markdown("---")
    st.markdown("**Session stats**")
    st.markdown(
        f"""
        <div class="stat-chip"><span>Queries</span><span>{st.session_state.stats['queries']}</span></div>
        <div class="stat-chip"><span>Tools run</span><span>{st.session_state.stats['tools_run']}</span></div>
        <div class="stat-chip"><span>Tools skipped</span><span>{st.session_state.stats['tools_skipped']}</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.display_log = []
        st.session_state.pending_tool_calls = None
        st.rerun()

    st.markdown("---")
    st.caption("Needs `GROQ_API_KEY`, `TAVILY_API_KEY`, `OPENWEATHER_API_KEY` in your `.env`.")

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <div class="badge">🛰️</div>
        <div>
            <h1>City Intelligence System</h1>
            <p>Ask about weather or news for any city — the agent decides which tools to call.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Render chat history
# ────────────────────────────────────────────────────────────────────────────
for entry in st.session_state.display_log:
    role = entry["role"]
    with st.chat_message(role, avatar="🛰️" if role == "assistant" else "🧑"):
        if entry["kind"] == "text":
            st.markdown(entry["content"])
        elif entry["kind"] == "tool_result":
            icon = {"get_weather": "🌤️", "get_news": "📰"}.get(entry["tool_name"], "🔧")
            label = {"get_weather": "Weather", "get_news": "News"}.get(entry["tool_name"], entry["tool_name"])
            st.markdown(f"<div class='tool-label'>{icon} {label}</div>", unsafe_allow_html=True)

            if entry["tool_name"] == "get_news" and entry.get("articles"):
                for a in entry["articles"]:
                    st.markdown(
                        f"""
                        <div class="news-card">
                            <div class="news-title">{a['title']}</div>
                            <div class="news-snippet">{a['snippet']}</div>
                            <div class="news-domain">{a['domain']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if a.get("url"):
                        st.link_button("Read article →", a["url"], use_container_width=False)
            else:
                st.markdown(
                    f"""<div class="tool-card"><div class="tool-body">{entry['content']}</div></div>""",
                    unsafe_allow_html=True,
                )
        elif entry["kind"] == "tool_skipped":
            st.markdown(
                f"""
                <div class="confirm-card">
                    <div class="confirm-title">⏭️ Skipped: {entry['tool_name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ────────────────────────────────────────────────────────────────────────────
# Core: run one LLM turn, stop if tool confirmation is needed
# ────────────────────────────────────────────────────────────────────────────
def run_llm_turn():
    """Invoke the LLM. If it wants to call tools, stash them for confirmation
    and return. Otherwise stream the final text answer."""
    result = llm_with_tools.invoke(st.session_state.messages)
    st.session_state.messages.append(result)

    if result.tool_calls:
        st.session_state.pending_tool_calls = list(result.tool_calls)
    else:
        st.session_state.display_log.append(
            {"role": "assistant", "kind": "text", "content": result.content}
        )


def execute_tool_call(tool_call, approved: bool):
    """Run (or skip) a single tool call, append the result, and continue
    the agent loop."""
    tool_name = tool_call["name"]

    if not approved:
        st.session_state.stats["tools_skipped"] += 1
        st.session_state.display_log.append(
            {"role": "assistant", "kind": "tool_skipped", "tool_name": tool_name}
        )
        # Tell the model the tool was declined so it can respond sensibly
        st.session_state.messages.append(
            ToolMessage(
                content=f"User declined to run {tool_name}.",
                tool_call_id=tool_call["id"],
            )
        )
        return

    tool_result = TOOLS[tool_name].invoke(tool_call)
    st.session_state.stats["tools_run"] += 1

    content = tool_result.content if isinstance(tool_result, ToolMessage) else tool_result
    log_entry = {"role": "assistant", "kind": "tool_result", "tool_name": tool_name, "content": content}

    if tool_name == "get_news":
        log_entry["articles"] = st.session_state.pop("_last_news_articles", [])

    st.session_state.display_log.append(log_entry)
    st.session_state.messages.append(
        tool_result if isinstance(tool_result, ToolMessage) else ToolMessage(
            content=content, tool_call_id=tool_call["id"]
        )
    )


# ────────────────────────────────────────────────────────────────────────────
# Pending tool-call confirmation UI (human in the loop)
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.pending_tool_calls:
    call = st.session_state.pending_tool_calls[0]
    with st.chat_message("assistant", avatar="🛰️"):
        st.markdown(
            f"""
            <div class="confirm-card">
                <div class="confirm-title">⚠️ Approval needed — {call['name']}</div>
                <div class="confirm-args">args: {call['args']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        approve = c1.button("✅ Approve", key=f"approve_{call['id']}", use_container_width=True)
        deny = c2.button("🚫 Deny", key=f"deny_{call['id']}", use_container_width=True)

    if approve or deny:
        execute_tool_call(call, approved=approve)
        st.session_state.pending_tool_calls.pop(0)

        if not st.session_state.pending_tool_calls:
            st.session_state.pending_tool_calls = None
            run_llm_turn()

        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Chat input
# ────────────────────────────────────────────────────────────────────────────
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

if st.session_state.pending_tool_calls is None:
    if st.session_state.awaiting_response:
        # The question is already rendered above (it's in display_log).
        # Now compute the answer, in its own render pass, with a spinner.
        with st.chat_message("assistant", avatar="🛰️"):
            with st.spinner("Thinking…"):
                run_llm_turn()
        st.session_state.awaiting_response = False
        st.rerun()
    else:
        user_input = st.chat_input("Ask about weather or news in a city…")
        if user_input:
            st.session_state.stats["queries"] += 1
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.display_log.append(
                {"role": "user", "kind": "text", "content": user_input}
            )
            st.session_state.awaiting_response = True
            st.rerun()