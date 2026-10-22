"""Visual system for the Quickr dashboard.

The palette is inspired by quick-commerce branding (yellow highlight, deep
green, warm off-white) without copying any proprietary logo or asset.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

COLORS = {
    "yellow": "#F8CB46",
    "yellow_soft": "#FDE9A8",
    "green": "#0B3D2C",
    "green_2": "#145A40",
    "sage": "#7BAE8B",
    "off_white": "#FAF7EF",
    "cream": "#FFF9E9",
    "charcoal": "#1F2A26",
    "muted": "#6F786F",
    "border": "#E7DFCF",
    "red": "#B5473A",
    "blue": "#4F7CAC",
}
COLORWAY = [
    COLORS["green"],
    COLORS["yellow"],
    COLORS["sage"],
    "#C97B4A",
    "#4F7CAC",
    "#9B8AC4",
]
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --qr-yellow: #F8CB46;
    --qr-yellow-soft: #FDE9A8;
    --qr-green: #0B3D2C;
    --qr-green-2: #145A40;
    --qr-sage: #7BAE8B;
    --qr-offwhite: #FAF7EF;
    --qr-cream: #FFF9E9;
    --qr-charcoal: #1F2A26;
    --qr-muted: #6F786F;
    --qr-border: #E7DFCF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
}
.stApp {
    background: var(--qr-offwhite);
    color: var(--qr-charcoal);
}
header[data-testid="stHeader"] {
    background: transparent;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3.5rem;
    max-width: 1380px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--qr-green);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}
section[data-testid="stSidebar"] * {
    color: #F7F3E8;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.14);
    margin: 1rem 0;
}
.qr-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 0.1rem 0.2rem 1.15rem 0.2rem;
}
.qr-brand-mark {
    width: 46px; height: 46px; border-radius: 13px;
    background: var(--qr-yellow); color: var(--qr-green);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem;
    font-weight: 700; letter-spacing: -0.06em;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}
.qr-brand-word {
    font-family: 'Space Grotesk', sans-serif;
    color: #FFF9E9; font-size: 1.62rem; font-weight: 700;
    line-height: 1.05; letter-spacing: -0.035em;
}
.qr-brand-tag {
    color: var(--qr-yellow); font-size: 0.56rem; font-weight: 700;
    letter-spacing: 0.19em; text-transform: uppercase; margin-top: 4px;
}
.qr-side-label {
    color: rgba(255,249,233,0.55) !important;
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; margin: 0.2rem 0 0.35rem 0.2rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.18rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 10px;
    padding: 0.48rem 0.55rem;
    margin: 0;
    transition: background 120ms ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--qr-yellow);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: var(--qr-green) !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 0.9rem;
    color: #F7F3E8;
}
section[data-testid="stSidebar"] div[role="radiogroup"] input {
    accent-color: var(--qr-green);
}
.qr-side-status {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; font-size: 0.76rem; color: rgba(255,249,233,0.85) !important;
    padding: 0.36rem 0.15rem;
}
.qr-side-status strong {color: #FFF9E9 !important; font-weight: 600;}
.qr-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #6BD99B; display: inline-block; margin-right: 6px;
    box-shadow: 0 0 0 3px rgba(107,217,155,0.12);
}
.qr-dot--off {
    background: #E27A6C; box-shadow: 0 0 0 3px rgba(226,122,108,0.14);
}
.qr-side-note {
    font-size: 0.68rem; line-height: 1.5; color: rgba(255,249,233,0.5) !important;
    margin-top: 0.7rem;
}

/* ---------- Typography ---------- */
.qr-eyebrow {
    color: var(--qr-green-2); font-size: 0.69rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.qr-page-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--qr-green); font-size: clamp(1.75rem, 3vw, 2.4rem);
    font-weight: 700; line-height: 1.08; letter-spacing: -0.045em;
    margin: 0 0 0.4rem 0;
}
.qr-page-sub {
    color: var(--qr-muted); font-size: 0.94rem; line-height: 1.6;
    max-width: 760px; margin-bottom: 1.25rem;
}
.qr-section-title {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--qr-green); font-size: 1.12rem; font-weight: 700;
    letter-spacing: -0.02em; margin: 1.3rem 0 0.2rem 0;
}
.qr-section-caption {
    color: var(--qr-muted); font-size: 0.8rem; margin-bottom: 0.6rem;
}

/* ---------- Cards ---------- */
.qr-kpi {
    background: #FFFFFF;
    border: 1px solid var(--qr-border);
    border-radius: 17px;
    padding: 16px 18px 15px 18px;
    min-height: 116px;
    box-shadow: 0 2px 5px rgba(31,42,38,0.035);
}
.qr-kpi--yellow { border-top: 4px solid var(--qr-yellow); }
.qr-kpi--green { border-top: 4px solid var(--qr-green); }
.qr-kpi-label {
    color: var(--qr-muted); font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.11em; text-transform: uppercase; line-height: 1.3;
    min-height: 1.8em;
}
.qr-kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--qr-green); font-size: 1.72rem; font-weight: 700;
    letter-spacing: -0.035em; line-height: 1.12; margin: 0.35rem 0 0.2rem 0;
}
.qr-kpi-note {
    color: var(--qr-muted); font-size: 0.73rem; line-height: 1.35;
}
.qr-insight {
    background: var(--qr-cream);
    border: 1px solid #F1E5C9;
    border-left: 4px solid var(--qr-yellow);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 9px;
    color: var(--qr-charcoal);
    font-size: 0.84rem;
    line-height: 1.5;
}
.qr-insight strong {
    display: block; color: var(--qr-green); font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px;
}
.qr-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #EAF3EC; color: #0B5D3B;
    border: 1px solid #D3E6D8;
    border-radius: 999px; padding: 3px 9px;
    font-size: 0.67rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; margin-right: 4px;
}
.qr-badge--yellow {
    background: #FFF4CC; color: #6B4E00; border-color: #F3D97C;
}
.qr-badge--red {
    background: #FBEAE7; color: #98372B; border-color: #F0CFC8;
}
.qr-result {
    background: var(--qr-green);
    border-radius: 20px;
    padding: 24px 26px 22px 26px;
    color: #FFF9E9;
    min-height: 238px;
    box-shadow: 0 10px 30px rgba(11,61,44,0.14);
}
.qr-result-label {
    color: var(--qr-yellow); font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
}
.qr-result-value {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--qr-yellow); font-size: 3.35rem; font-weight: 700;
    letter-spacing: -0.05em; line-height: 1.05; margin: 0.45rem 0 0.1rem 0;
}
.qr-result-unit {color: rgba(255,249,233,0.7); font-size: 0.88rem;}
.qr-result-row {
    display: flex; justify-content: space-between; gap: 12px;
    border-top: 1px solid rgba(255,255,255,0.14);
    padding-top: 10px; margin-top: 12px;
    font-size: 0.78rem; color: rgba(255,249,233,0.8);
}
.qr-result-row strong {color: #FFF9E9; font-weight: 600;}
.qr-callout {
    background: #FFFFFF; border: 1px solid var(--qr-border);
    border-radius: 13px; padding: 13px 15px; color: var(--qr-charcoal);
    font-size: 0.84rem; line-height: 1.55;
}
.qr-callout strong {color: var(--qr-green);}
.qr-source {
    color: var(--qr-muted); font-size: 0.74rem; line-height: 1.5;
}

/* ---------- Widgets ---------- */
.stButton > button, .stFormSubmitButton > button {
    background: var(--qr-yellow);
    color: var(--qr-green);
    border: 1px solid #E9BC2F;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.48rem 1.05rem;
    transition: transform 80ms ease, background 120ms ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #FFD75E;
    color: var(--qr-green);
    border-color: #E0B126;
    transform: translateY(-1px);
}
.stButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(0);
}
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid var(--qr-border);
    border-radius: 18px;
    padding: 1.1rem 1.2rem 0.6rem 1.2rem;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    border-radius: 9px !important;
    border-color: #DDD5C4 !important;
    background: #FFFEFA !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--qr-green) !important;
    box-shadow: 0 0 0 1px var(--qr-green) !important;
}
label, [data-testid="stWidgetLabel"] p {
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    color: #4C5952 !important;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--qr-border);
    border-radius: 13px;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--qr-border);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid var(--qr-border);
    border-radius: 17px;
    padding: 10px 10px 4px 10px;
    box-shadow: 0 2px 5px rgba(31,42,38,0.03);
}
@media (max-width: 900px) {
    .block-container {padding-left: 1rem; padding-right: 1rem;}
    .qr-page-title {font-size: 1.8rem;}
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        """
        <div class="qr-brand">
            <div class="qr-brand-mark">Q</div>
            <div>
                <div class="qr-brand-word">Quickr</div>
                <div class="qr-brand-tag">Delivery intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def side_label(text: str) -> None:
    st.markdown(f'<div class="qr-side-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def status_row(label: str, value: str, ok: bool = True) -> None:
    dot = "qr-dot" if ok else "qr-dot qr-dot--off"
    st.markdown(
        f'<div class="qr-side-status"><span><span class="{dot}"></span>'
        f"{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>",
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = "".join(
        f'<span class="qr-badge">{html.escape(item)}</span>' for item in (badges or [])
    )
    st.markdown(
        f"""
        <div class="qr-eyebrow">{html.escape(eyebrow)}</div>
        <div class="qr-page-title">{html.escape(title)}</div>
        <div class="qr-page-sub">{html.escape(subtitle)}</div>
        <div style="margin:-0.6rem 0 1.2rem 0;">{badge_html}</div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="qr-section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="qr-section-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, note: str, accent: str = "yellow") -> None:
    accent_class = "qr-kpi--green" if accent == "green" else "qr-kpi--yellow"
    st.markdown(
        f"""
        <div class="qr-kpi {accent_class}">
            <div class="qr-kpi-label">{html.escape(label)}</div>
            <div class="qr-kpi-value">{html.escape(value)}</div>
            <div class="qr-kpi-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(title: str, text: str) -> None:
    st.markdown(
        f'<div class="qr-insight"><strong>{html.escape(title)}</strong>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def callout(text_html: str) -> None:
    st.markdown(f'<div class="qr-callout">{text_html}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "green") -> str:
    cls = "qr-badge qr-badge--yellow" if tone == "yellow" else (
        "qr-badge qr-badge--red" if tone == "red" else "qr-badge"
    )
    return f'<span class="{cls}">{html.escape(text)}</span>'


def style_fig(fig: go.Figure, height: int = 300, show_legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=38, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color=COLORS["charcoal"], size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=15, color=COLORS["green"]), x=0.02, xanchor="left"),
        colorway=COLORWAY,
        showlegend=show_legend,
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=COLORS["green"], font=dict(color="#FFF9E9", size=12), bordercolor=COLORS["green"]),
        transition=dict(duration=0),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#EEE7D8", zeroline=False,
        linecolor="#DED5C4", tickfont=dict(size=11, color=COLORS["muted"]),
        title_font=dict(size=11, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EEE7D8", zeroline=False,
        linecolor="#DED5C4", tickfont=dict(size=11, color=COLORS["muted"]),
        title_font=dict(size=11, color=COLORS["muted"]),
    )
    return fig
