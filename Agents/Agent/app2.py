"""
Atlas — City Assistant (Streamlit UI, plain LangChain, no agent framework)

Deliberately simple. No create_agent, no middleware, no LangGraph, no
background threads. Just:

    1. Ask the model, with tools bound, what it wants to do.
    2. If it wants to call a tool, show an Approve/Deny button for it.
    3. Once approved, call the tool directly (a normal Python function
       call, nothing fancy) and record the result.
    4. Send the result back to the model and show its final reply.

Everything happens on Streamlit's main thread, so there's nothing for
st.session_state to lose track of, and no "missing ScriptRunContext"
warnings.

Setup:
    pip install -r requirements.txt
    # copy .env.example to .env and fill in your real keys
    streamlit run streamlit_app.py
"""

import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

st.set_page_config(page_title="Atlas — City Assistant", page_icon="🧭", layout="wide")

# ==========================================================================
# Theme
# ==========================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --bg: #12141a; --surface: #1b1e27; --surface-2: #232733; --border: #2c313d;
  --text: #e7e9ee; --text-dim: #9aa1b0; --text-faint: #6b7180;
  --copper: #c98a4b; --cyan: #6fcbd8; --violet: #8e8cf0; --amber: #e0b34c;
  --ok: #6fd89a; --err: #e8807f;
}
html, body, [class*="css"]{ font-family: 'IBM Plex Sans', sans-serif; }
.stApp{ background: var(--bg); color: var(--text); }
#MainMenu, footer, header{ visibility: hidden; }

section[data-testid="stSidebar"]{ background: #0e1015; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .block-container{ padding-top: 1.6rem; }

.brand-row{ display:flex; align-items:center; gap:12px; margin-bottom: 4px; }
.brand-mark{ width:36px; height:36px; border-radius:9px; flex-shrink:0;
  background: linear-gradient(155deg, var(--copper), #8a5a2c);
  display:flex; align-items:center; justify-content:center;
  font-family:'Fraunces', serif; font-weight:600; font-size:16px; color:#161311; }
.brand-title{ font-family:'Fraunces', serif; font-weight:600; font-size:19px; color: var(--text); line-height:1.1; }
.brand-sub{ font-size:12px; color: var(--text-faint); margin-top:2px; }

.sidebar-label{ font-size:11px; text-transform:uppercase; letter-spacing:.08em;
  color: var(--text-faint); margin: 18px 0 8px 0; font-weight:500; }

.tool-card{ border:1px solid var(--border); background: var(--surface);
  border-radius:10px; padding:10px 12px; margin-bottom:8px; }
.tool-name{ font-family:'IBM Plex Mono', monospace; font-size:12.5px; color: var(--text); }
.tool-desc{ font-size:11.5px; color: var(--text-faint); margin-top:3px; line-height:1.45; }

.runtime-box{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; color: var(--text-faint);
  line-height:1.9; border-top:1px solid var(--border); padding-top:14px; margin-top:16px; }
.runtime-box b{ color: var(--text-dim); font-weight:500; }

.key-pill{ display:inline-flex; align-items:center; gap:6px; font-size:12px;
  padding:6px 10px; border-radius:8px; border:1px solid var(--border);
  font-family:'IBM Plex Mono', monospace; }
.key-pill.ok{ color: var(--ok); border-color: rgba(111,216,154,0.3); }
.key-pill.bad{ color: var(--err); border-color: rgba(232,128,127,0.3); }

.hero{ padding: 4px 0 18px 0; border-bottom: 1px solid var(--border); margin-bottom: 22px; }
.hero h1{ font-family:'Fraunces', serif; font-weight:600; font-size: 30px; margin:0; color: var(--text); }
.hero p{ color: var(--text-dim); font-size: 14px; margin: 6px 0 0 0; }

div[data-testid="stChatMessage"]{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 4px 6px; margin-bottom: 4px; }

div[data-testid="stChatInput"] textarea{ font-family:'IBM Plex Sans', sans-serif;
  background: var(--surface) !important; color: var(--text) !important; border-radius: 10px !important; }
div[data-testid="stChatInput"]{ border-top: 1px solid var(--border); }

.stButton>button{ background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; }
.stButton>button:hover{ border-color: var(--copper); color: var(--text); }

.result-card{ border-radius: 12px; border: 1px solid var(--border); background: var(--surface);
  padding: 15px 17px; margin: 10px 0; }
.result-card.weather{ border-top: 2px solid var(--cyan); }
.result-card.news{ border-top: 2px solid var(--violet); }
.result-label{ font-family:'IBM Plex Mono', monospace; font-size:10.5px; text-transform:uppercase;
  letter-spacing:.07em; color: var(--text-faint); margin-bottom:10px; display:flex; align-items:center; gap:7px; }
.result-label .sw{ width:6px; height:6px; border-radius:50%; }
.weather .sw{ background: var(--cyan); }
.news .sw{ background: var(--violet); }

.wx-temp{ font-family:'Fraunces', serif; font-size:32px; font-weight:500; line-height:1; margin-bottom:4px; }
.wx-desc{ font-size:13.5px; color: var(--text-dim); text-transform:capitalize; margin-bottom:12px; }
.wx-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; font-size:12.5px; }
.wx-grid span{ display:block; color: var(--text-faint); font-size:10.5px; margin-bottom:2px; }

.news-item{ padding: 9px 0; border-bottom: 1px solid var(--border); }
.news-item:last-child{ border-bottom:none; padding-bottom:0; }
.news-title a{ color: var(--text); text-decoration:none; font-size:13.5px; line-height:1.4; }
.news-title a:hover{ color: var(--copper); }

.tool-trace-line{ font-family:'IBM Plex Mono', monospace; font-size:12px; color: var(--text-dim);
  background: var(--surface); border: 1px solid var(--border); border-left: 2px solid var(--copper);
  padding: 7px 11px; border-radius: 7px; margin-bottom: 6px; }

.denied-note{ font-size: 13px; color: var(--err); padding: 8px 2px; }

.approval-card{ border: 1px solid var(--border); border-left: 3px solid var(--amber);
  background: var(--surface); border-radius: 10px; padding: 13px 15px; margin: 6px 0 12px; }
.approval-title{ font-family:'IBM Plex Mono', monospace; font-size:12.5px; color: var(--amber); margin-bottom:2px; }
.approval-sub{ font-size:12.5px; color: var(--text-dim); margin-bottom:10px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================================
# Tools — same as your script, called directly (no agent executor)
# ==========================================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of the city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather tool is not configured: OPENWEATHER_API_KEY is missing."

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=8)
        data = response.json()
    except requests.RequestException as e:
        return f"Could not reach the weather service: {e}"

    if response.status_code != 200:
        return f"Could not get weather for {city}. Error: {data.get('message', 'Unknown error')}"

    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels = data["main"].get("feels_like", temp)
    humidity = data["main"].get("humidity", "?")

    return (
        f"The weather in {city}: {description}, {temp}°C "
        f"(feels like {feels}°C), humidity {humidity}%."
    )


_tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=_tavily_key) if _tavily_key else None


@tool
def get_news(city: str) -> str:
    """Get the latest news about a city"""
    if tavily_client is None:
        return "News tool is not configured: TAVILY_API_KEY is missing."

    query = f"Get the top 5 latest news about the {city}"
    try:
        response = tavily_client.search(
            query=query, search_depth="basic", max_results=5, topic="news"
        )
    except Exception as e:
        return f"Could not reach the news service: {e}"

    data = response.get("results")
    if not data:
        return f"No news found for {city}."

    news_list = [f"- {item.get('title')} ({item.get('url')})" for item in data]
    return f"Latest news in {city}:\n" + "\n".join(news_list)


TOOLS_BY_NAME = {"get_weather": get_weather, "get_news": get_news}

SYSTEM_PROMPT = (
    "You are a helpful, unbiased, and truthful city assistant. "
    "You can look up current weather and news for cities in India using your tools. "
    "If the user greets you or is just chatting, respond naturally without calling any tool. "
    "If they ask about weather or news but do not name a specific city, ask them which city "
    "they mean instead of guessing one or calling a tool with a blank city."
)


@st.cache_resource
def get_llm():
    llm = ChatGroq(model="openai/gpt-oss-20b")
    return llm.bind_tools([get_weather, get_news])


def keys_present() -> dict:
    return {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "OPENWEATHER_API_KEY": bool(os.getenv("OPENWEATHER_API_KEY")),
    }


# ==========================================================================
# Parsers — real tool output -> rich cards
# ==========================================================================

def parse_weather(text: str):
    m = re.search(
        r"weather in .+?:\s*(.+?),\s*(-?[\d.]+)°C\s*\(feels like (-?[\d.]+)°C\),\s*humidity (\S+)%",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {"desc": m.group(1), "temp": m.group(2), "feels": m.group(3), "humidity": m.group(4)}


def parse_news(text: str):
    items = []
    for line in text.splitlines():
        m = re.match(r"-\s*(.+?)\s*\((https?://\S+)\)\s*$", line.strip())
        if m:
            items.append({"title": m.group(1), "url": m.group(2)})
    return items


def render_weather_card(city: str, raw: str) -> str:
    data = parse_weather(raw)
    if not data:
        return f'<div class="result-card weather"><div class="result-label"><span class="sw"></span>get_weather → {city}</div>{raw}</div>'
    return f"""
    <div class="result-card weather">
      <div class="result-label"><span class="sw"></span>get_weather → {city}</div>
      <div class="wx-temp">{data['temp']}°C</div>
      <div class="wx-desc">{data['desc']}</div>
      <div class="wx-grid">
        <div><span>Feels like</span>{data['feels']}°C</div>
        <div><span>Humidity</span>{data['humidity']}%</div>
      </div>
    </div>
    """


def render_news_card(city: str, raw: str) -> str:
    items = parse_news(raw)
    if not items:
        return f'<div class="result-card news"><div class="result-label"><span class="sw"></span>get_news → {city}</div>{raw}</div>'
    rows = "".join(
        f'<div class="news-item"><div class="news-title"><a href="{it["url"]}" target="_blank">{it["title"]}</a></div></div>'
        for it in items
    )
    return f'<div class="result-card news"><div class="result-label"><span class="sw"></span>get_news → {city}</div>{rows}</div>'


def render_tool_result(tc: dict):
    city = tc["args"].get("city", "")
    st.markdown(f'<div class="tool-trace-line">{tc["name"]}(city="{city}")</div>', unsafe_allow_html=True)
    result = tc.get("result") or ""
    if "denied by user" in result.lower():
        st.markdown('<div class="denied-note">🚫 Denied — this tool call was not approved.</div>', unsafe_allow_html=True)
    elif tc["name"] == "get_weather":
        st.markdown(render_weather_card(city, result), unsafe_allow_html=True)
    elif tc["name"] == "get_news":
        st.markdown(render_news_card(city, result), unsafe_allow_html=True)


# ==========================================================================
# Session state
#
#   messages       finalized chat turns, for display
#   active_query   the user text currently mid-flight, or None when idle
#   pending_calls  list of {id, name, args, status} for the current turn —
#                  status is "pending" / "approved" / "denied"
#   tool_results   {call_id: result_text} once a call has been run or denied
# ==========================================================================

for key, default in [
    ("messages", []),
    ("active_query", None),
    ("pending_calls", None),
    ("tool_results", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_turn_state():
    st.session_state.pending_calls = None
    st.session_state.tool_results = {}


# ==========================================================================
# Sidebar
# ==========================================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">A</div>
          <div>
            <div class="brand-title">Atlas</div>
            <div class="brand-sub">City intelligence agent</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Tool registry</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tool-card">
          <div class="tool-name">get_weather(city)</div>
          <div class="tool-desc">Live conditions via OpenWeather. Gated by approval.</div>
        </div>
        <div class="tool-card">
          <div class="tool-name">get_news(city)</div>
          <div class="tool-desc">Headlines via Tavily search. Gated by approval.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Connection</div>', unsafe_allow_html=True)
    for name, ok in keys_present().items():
        cls = "ok" if ok else "bad"
        icon = "●" if ok else "○"
        st.markdown(f'<div class="key-pill {cls}">{icon} {name}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="runtime-box">
          model: <b>openai/gpt-oss-20b</b><br>
          provider: <b>groq</b><br>
          framework: <b>langchain (bind_tools)</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_query = None
        reset_turn_state()
        st.rerun()

# ==========================================================================
# Header
# ==========================================================================

st.markdown(
    """
    <div class="hero">
      <h1>City Assistant</h1>
      <p>Ask about weather, news, or both — for any city. Every tool call needs your approval first.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Chat history
# ==========================================================================

for msg in st.session_state.messages:
    avatar = "🧭" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        for tc in msg.get("tool_calls", []):
            render_tool_result(tc)
        st.markdown(msg["content"])

# ==========================================================================
# Step 1 — ask the model what it wants to do (only if we don't already
# have a proposal for this turn)
# ==========================================================================

if st.session_state.active_query and st.session_state.pending_calls is None:
    if not keys_present()["GROQ_API_KEY"]:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "GROQ_API_KEY is missing — check your .env file.",
                "tool_calls": [],
            }
        )
        st.session_state.active_query = None
        st.rerun()
    else:
        with st.spinner("Thinking…"):
            llm = get_llm()
            try:
                ai_response = llm.invoke(
                    [SystemMessage(SYSTEM_PROMPT), HumanMessage(st.session_state.active_query)]
                )
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"The model call failed: {e}", "tool_calls": []}
                )
                st.session_state.active_query = None
                st.rerun()

            tool_calls = getattr(ai_response, "tool_calls", None) or []

            if not tool_calls:
                # plain conversational reply, nothing to approve
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response.content, "tool_calls": []}
                )
                st.session_state.active_query = None
            else:
                st.session_state.pending_calls = [
                    {"id": tc["id"], "name": tc["name"], "args": tc.get("args", {}), "status": "pending"}
                    for tc in tool_calls
                ]

        st.rerun()

# ==========================================================================
# Step 2 — walk through any pending tool calls, one approval card at a time
# ==========================================================================

if st.session_state.pending_calls:
    next_pending = next((pc for pc in st.session_state.pending_calls if pc["status"] == "pending"), None)

    if next_pending:
        city = next_pending["args"].get("city", "")
        with st.chat_message("assistant", avatar="🧭"):
            st.markdown(
                f"""
                <div class="approval-card">
                  <div class="approval-title">⏸ awaiting approval — {next_pending['name']}(city="{city}")</div>
                  <div class="approval-sub">The agent wants to call this tool. Approve to let it run, or deny to block it.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", key=f"approve_{next_pending['id']}", use_container_width=True):
                    tool_fn = TOOLS_BY_NAME[next_pending["name"]]
                    try:
                        result = tool_fn.invoke(next_pending["args"])
                    except Exception as e:
                        result = f"Tool call failed: {e}"
                    st.session_state.tool_results[next_pending["id"]] = result
                    next_pending["status"] = "approved"
                    st.rerun()
            with col2:
                if st.button("🚫 Deny", key=f"deny_{next_pending['id']}", use_container_width=True):
                    st.session_state.tool_results[next_pending["id"]] = "Tool call denied by user."
                    next_pending["status"] = "denied"
                    st.rerun()

    else:
        # every pending call has a decision — send results back and get
        # the model's final natural-language reply
        with st.spinner("Thinking…"):
            llm = get_llm()
            tool_messages = [
                ToolMessage(content=st.session_state.tool_results[pc["id"]], tool_call_id=pc["id"])
                for pc in st.session_state.pending_calls
            ]
            try:
                # Rebuild the AIMessage-with-tool-calls shape the model
                # needs to see before its own tool results.
                from langchain_core.messages import AIMessage as _AIMessage

                ai_msg_stub = _AIMessage(
                    content="",
                    tool_calls=[
                        {"id": pc["id"], "name": pc["name"], "args": pc["args"]}
                        for pc in st.session_state.pending_calls
                    ],
                )
                final_response = llm.invoke(
                    [
                        SystemMessage(SYSTEM_PROMPT),
                        HumanMessage(st.session_state.active_query),
                        ai_msg_stub,
                        *tool_messages,
                    ]
                )
                final_reply = final_response.content
            except Exception as e:
                final_reply = f"The model call failed: {e}"

            tool_calls_log = [
                {
                    "name": pc["name"],
                    "args": pc["args"],
                    "result": st.session_state.tool_results.get(pc["id"], ""),
                }
                for pc in st.session_state.pending_calls
            ]

            st.session_state.messages.append(
                {"role": "assistant", "content": final_reply, "tool_calls": tool_calls_log}
            )
            st.session_state.active_query = None
            reset_turn_state()

        st.rerun()

# ==========================================================================
# Chat input — disabled while a turn is in flight
# ==========================================================================

busy = st.session_state.active_query is not None
user_input = st.chat_input("Ask Atlas about a city…", disabled=busy)

if user_input and not busy:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.active_query = user_input
    reset_turn_state()
    st.rerun()