"""
app.py — GazeSpeak Streamlit launcher.

Provides a polished landing page, patient setup, and live session dashboard.
The actual eye-tracking runs as a separate pygame process launched from here.

Run:
    streamlit run app.py
"""

import streamlit as st
import subprocess, json, sys, time, datetime
from pathlib import Path

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title  = "GazeSpeak — AAC System",
    page_icon   = "👁️",
    layout      = "centered",
    initial_sidebar_state = "collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR    = Path(__file__).parent
STATUS_FILE = WORK_DIR / "session_status.json"
LOG_FILE    = WORK_DIR / "medical_log.json"
PYTHON      = sys.executable
URGENT      = {"PAIN", "CALL", "TOILET"}

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Chrome removal ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; max-width: 700px !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Theme tokens ─────────────────────────────────────────────────── */
:root {
    --bg:       #07101f;
    --card:     #0d1a2e;
    --card2:    #112240;
    --accent:   #00c8f0;
    --purple:   #7c3aed;
    --green:    #10b981;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   rgba(0,200,240,0.14);
    --shadow:   0 20px 60px rgba(0,0,0,0.6);
}

html, body, .stApp, .main { background: var(--bg) !important; color: var(--text) !important; }

/* ── Animations ───────────────────────────────────────────────────── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes pulseDot { 0%,100%{transform:scale(0.9);opacity:.5} 50%{transform:scale(1.1);opacity:1} }
@keyframes spin     { to{transform:rotate(360deg)} }
@keyframes shimmer  {
    0%  { background-position: -400px 0; }
    100%{ background-position:  400px 0; }
}

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 80px 24px 56px;
    animation: fadeUp 1s ease-out;
}
.hero-eye {
    font-size: 72px;
    display: inline-block;
    animation: pulseDot 3s ease-in-out infinite;
    margin-bottom: 20px;
}
.hero-title {
    font-size: clamp(2.6rem,6vw,4rem);
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg,#fff 20%,var(--accent) 60%,var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 14px;
}
.hero-sub {
    font-size: 1.05rem; color: var(--muted); max-width: 480px;
    margin: 0 auto 36px; line-height: 1.75; text-align: center;
}
.badges { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-bottom:44px; }
.badge  {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 999px; padding: 5px 14px;
    font-size: 0.78rem; color: var(--accent); letter-spacing:.4px;
}
.divider-line {
    height:1px;
    background: linear-gradient(90deg,transparent,var(--border),transparent);
    margin: 0 0 44px;
}

/* ── Cards ────────────────────────────────────────────────────────── */
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 40px 44px 36px;
    box-shadow: var(--shadow); animation: fadeUp .7s ease-out;
    margin-bottom: 24px;
}
.card-title { font-size:1.55rem; font-weight:800; color:#fff; margin-bottom:6px; }
.card-sub   { font-size:.9rem; color:var(--muted); margin-bottom:24px; }

/* ── Input ────────────────────────────────────────────────────────── */
.stTextInput>div>div>input {
    background: var(--card2) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text) !important;
    font-size: 1.05rem !important; padding: 14px 18px !important;
}
.stTextInput>div>div>input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,200,240,.18) !important;
    outline: none !important;
}
.stTextInput label { color:var(--muted)!important; font-size:.82rem!important;
    font-weight:600!important; letter-spacing:.8px!important; text-transform:uppercase!important; }

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg, var(--accent), #0096b8) !important;
    color: #03111e !important; font-weight: 800 !important; font-size: 1rem !important;
    padding: 14px 32px !important; border-radius: 12px !important; border: none !important;
    width: 100% !important; letter-spacing: .4px !important;
    box-shadow: 0 4px 22px rgba(0,200,240,.35) !important;
    transition: all .18s ease !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,200,240,.5) !important;
}
.stButton>button:active { transform: translateY(0) !important; }

/* ── Greeting ─────────────────────────────────────────────────────── */
.greeting { text-align:center; padding: 60px 24px 16px; animation: fadeUp .9s ease-out; }
.greet-hello { font-size:.95rem; color:var(--muted); letter-spacing:4px;
    text-transform:uppercase; margin-bottom:8px; }
.greet-name  {
    font-size: clamp(2.8rem,7vw,4.5rem); font-weight:900;
    background: linear-gradient(135deg,#fff 30%,var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom:12px; letter-spacing:-1px;
}
.greet-msg { font-size:.95rem; color:var(--muted); margin-bottom:36px; }

/* ── Status ───────────────────────────────────────────────────────── */
.status-box {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 32px; text-align: center; margin-bottom: 24px; animation: fadeUp .6s ease-out;
}
.status-icon  { font-size:2.8rem; margin-bottom:12px; }
.status-title { font-size:1.25rem; font-weight:800; color:#fff; margin-bottom:6px; }
.status-sub   { font-size:.88rem; color:var(--muted); }

/* ── Live feed ────────────────────────────────────────────────────── */
.feed-wrap  { margin-top: 8px; }
.feed-label { font-size:.78rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
    color:var(--muted); margin-bottom:10px; }
.feed-item  {
    display:flex; justify-content:space-between; align-items:center;
    background: var(--card2); border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0; padding: 10px 16px; margin-bottom:7px;
    animation: fadeUp .35s ease-out;
}
.feed-item.urgent { border-left-color: var(--red); }
.feed-word { font-weight:800; font-size:1rem; color:#fff; }
.feed-time { font-size:.76rem; color:var(--muted); }
.feed-urgent-tag {
    background: rgba(239,68,68,.15); color:var(--red);
    border-radius:6px; padding:2px 8px; font-size:.72rem; font-weight:700;
}

/* ── Spinner dots ─────────────────────────────────────────────────── */
.dots { display:inline-flex; gap:5px; align-items:center; }
.dots span {
    width:7px; height:7px; border-radius:50%; background:var(--accent);
    animation: pulseDot 1.4s ease-in-out infinite;
}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}

/* ── Summary ──────────────────────────────────────────────────────── */
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:16px 0 24px; }
.stat-box  {
    background:var(--card2); border:1px solid var(--border); border-radius:14px;
    padding:20px; text-align:center;
}
.stat-num  { font-size:2rem; font-weight:900; color:var(--accent); }
.stat-lbl  { font-size:.78rem; color:var(--muted); margin-top:4px; }

</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────
def _init():
    defaults = {
        "page":          "landing",
        "patient_name":  "",
        "session_start": None,
        "proc":          None,   # subprocess.Popen object
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helpers ───────────────────────────────────────────────────────────────
def _read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {"state": "idle"}

def _read_log() -> list:
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return []

def _launch(patient_name: str) -> None:
    cmd = [PYTHON, str(WORK_DIR / "main_system.py"), "--flip",
           "--patient-name", patient_name]
    st.session_state.proc = subprocess.Popen(cmd, cwd=str(WORK_DIR))

def _stop() -> None:
    proc = st.session_state.proc
    if proc and proc.poll() is None:
        proc.terminate()
    st.session_state.proc = None


# ── Page renderers ────────────────────────────────────────────────────────

def page_landing():
    st.markdown("""
    <div class="hero">
        <div class="hero-eye">👁️</div>
        <div class="hero-title">GazeSpeak</div>
        <p class="hero-sub" style="text-align:center; display:block; width:100%; max-width:480px; margin:0 auto 36px;">
            An AI-powered eye-gaze communication system for ALS and paralysis
            patients — communicate with your eyes, no hands required.
        </p>
        
      
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "name"
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:48px; color:#2a3a52; font-size:.8rem; letter-spacing:.5px;">
        GRADUATION PROJECT · COMPUTER SCIENCE DEPARTMENT
    </div>
    """, unsafe_allow_html=True)


def page_name():
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.3, 2, 0.3])
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">👤 Patient Setup</div>
            <div class="card-sub">Enter the patient's name to personalise the session</div>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("PATIENT NAME", placeholder="e.g. Ahmed Hassan",
                             label_visibility="visible")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = "landing"
                st.rerun()
        with col_b:
            if st.button("Continue →", use_container_width=True):
                if name.strip():
                    st.session_state.patient_name = name.strip()
                    st.session_state.page = "greeting"
                    st.rerun()
                else:
                    st.error("Please enter the patient's name first.")


def page_greeting():
    name = st.session_state.patient_name
    st.markdown(f"""
    <div class="greeting">
        <div class="greet-hello">WELCOME BACK</div>
        <div class="greet-name">{name}</div>
        <div class="greet-msg">
            Everything is ready. When you click <strong>Start Session</strong>, the
            calibration window will open on your screen.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">📋 How it works</div>
        <div class="card-sub" style="margin-bottom:0">
            <ol style="padding-left:18px; line-height:2; color:#94a3b8; font-size:.92rem;">
                <li>A 9-point calibration grid appears — look at each red dot and press <strong>SPACE</strong></li>
                <li>The communication board opens automatically after calibration</li>
                <li>Hold your gaze on a word for ~0.8 s, then <strong>blink to confirm</strong></li>
                <li>Hold eyes closed for <strong>3 seconds</strong> to pause / resume the board</li>
                <li>Press <strong>R</strong> at any time to recalibrate | <strong>ESC</strong> to exit</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        if st.button("▶  Start Session", use_container_width=True):
            st.session_state.session_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _launch(st.session_state.patient_name)
            st.session_state.page = "running"
            st.rerun()


def page_running():
    name    = st.session_state.patient_name
    status  = _read_status()
    state   = status.get("state", "idle")

    # Check if subprocess finished
    proc = st.session_state.proc
    proc_done = (proc is not None and proc.poll() is not None)
    if state == "done" or proc_done:
        st.session_state.page = "done"
        st.rerun()

    # ── Status banner ──────────────────────────────────────────────
    if state == "calibrating":
        st.markdown("""
        <div class="status-box">
            <div class="status-icon">🎯</div>
            <div class="status-title">Calibrating…</div>
            <div class="status-sub">
                Look at each red dot and press <strong>SPACE</strong> to collect gaze samples.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-box" style="border-color:rgba(16,185,129,.3)">
            <div class="status-icon">✅</div>
            <div class="status-title">Session Active — {name}</div>
            <div class="status-sub">
                Board is running. Hold eyes closed 3 s to pause. Press ESC to end.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Live selection feed ────────────────────────────────────────
    log          = _read_log()
    session_start = st.session_state.session_start or "2000-01-01"
    session_log  = [e for e in log if e.get("time", "") >= session_start]

    if session_log:
        st.markdown('<div class="feed-wrap"><div class="feed-label">Live Selections</div>', unsafe_allow_html=True)
        for entry in reversed(session_log[-12:]):
            word    = entry.get("item", "")
            t       = entry.get("time", "")[-8:]
            urgent  = entry.get("urgent", False)
            urg_tag = '<span class="feed-urgent-tag">URGENT</span>' if urgent else ""
            cls     = "feed-item urgent" if urgent else "feed-item"
            st.markdown(f"""
            <div class="{cls}">
                <span class="feed-word">{word}</span>
                <span style="display:flex;gap:8px;align-items:center">
                    {urg_tag}
                    <span class="feed-time">{t}</span>
                </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; color:#2a3a52; padding:28px; font-size:.88rem;">
            Waiting for first selection…
            <div class="dots" style="margin-top:12px; justify-content:center;">
                <span></span><span></span><span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stop button ────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        if st.button("⏹  End Session", use_container_width=True):
            _stop()
            st.session_state.page = "done"
            st.rerun()

    # Poll every 2 s
    time.sleep(2)
    st.rerun()


def page_done():
    name          = st.session_state.patient_name
    session_start = st.session_state.session_start or "2000-01-01"
    log           = _read_log()
    session_log   = [e for e in log if e.get("time", "") >= session_start]
    urgent_count  = sum(1 for e in session_log if e.get("urgent"))

    # Duration
    try:
        t0  = datetime.datetime.strptime(session_start, "%Y-%m-%d %H:%M:%S")
        dur = str(datetime.datetime.now() - t0).split(".")[0]
    except Exception:
        dur = "—"

    st.markdown(f"""
    <div class="greeting">
        <div class="greet-hello">SESSION COMPLETE</div>
        <div class="greet-name" style="font-size:clamp(2rem,5vw,3rem)">{name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-num">{len(session_log)}</div>
            <div class="stat-lbl">Total Selections</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" style="color:{'#ef4444' if urgent_count else 'var(--accent)'}">
                {urgent_count}
            </div>
            <div class="stat-lbl">Urgent Requests</div>
        </div>
        <div class="stat-box">
            <div class="stat-num" style="font-size:1.4rem">{dur}</div>
            <div class="stat-lbl">Session Duration</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{session_start[11:16] if len(session_start)>11 else '—'}</div>
            <div class="stat-lbl">Started At</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if session_log:
        st.markdown('<div class="card"><div class="card-title">Session Log</div>', unsafe_allow_html=True)
        for entry in reversed(session_log):
            word   = entry.get("item", "")
            t      = entry.get("time", "")
            urgent = entry.get("urgent", False)
            cls    = "feed-item urgent" if urgent else "feed-item"
            urg    = '<span class="feed-urgent-tag">URGENT</span>' if urgent else ""
            st.markdown(f"""
            <div class="{cls}">
                <span class="feed-word">{word}</span>
                <span style="display:flex;gap:8px;align-items:center">
                    {urg}<span class="feed-time">{t[11:]}</span>
                </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        if st.button("＋  New Session", use_container_width=True):
            _stop()
            st.session_state.page         = "greeting"
            st.session_state.session_start = None
            st.rerun()


# ── Router ────────────────────────────────────────────────────────────────
page = st.session_state.page
if   page == "landing":  page_landing()
elif page == "name":     page_name()
elif page == "greeting": page_greeting()
elif page == "running":  page_running()
elif page == "done":     page_done()
