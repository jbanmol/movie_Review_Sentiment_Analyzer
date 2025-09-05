
# Streamlit web interface for the sentiment analyzer
import streamlit as st
import pandas as pd
import json
import time
import io
from sentiment_llm import analyze_sentiment, process_batch_reviews
import base64

# Configure the web app appearance and behavior
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling to make the app look polished
st.markdown("""
<style>
    /* Main container and background styling */
    .stApp { background-color: #000000; color: #ffffff; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; background-color: #000000; }

    /* Header styling */
    .main-header { text-align: center; margin-bottom: 2rem; }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem; text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); }
    .main-subtitle { font-size: 1.1rem; color: #b0b0b0; margin-bottom: 2rem; }

    /* Button styling */
    .stButton > button {
        background-color: #0A0A0A;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1a1a1a;
        border-color: #555555;
        box-shadow: 0 2px 8px rgba(255, 255, 255, 0.1);
    }
    .primary-button > button {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.3);
    }
    .primary-button > button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 20px rgba(37, 99, 235, 0.5);
    }

    /* Text area styling */
    .stTextArea textarea {
        background-color: #0a0a0a !important;
        border: 1px solid #444444 !important;
        border-radius: 8px;
        color: #ffffff !important;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus { border-color: #2563eb !important; box-shadow: 0 0 10px rgba(37, 99, 235, 0.3) !important; }

    /* Form containers */
    .stForm { background-color: transparent; border: none; }

    /* Results styling */
    .result-container {
        background-color: #0a0a0a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333333;
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.05);
    }
    .sentiment-positive { color: #22c55e; font-weight: 600; text-shadow: 0 0 5px rgba(34, 197, 94, 0.3); }
    .sentiment-negative { color: #ef4444; font-weight: 600; text-shadow: 0 0 5px rgba(239, 68, 68, 0.3); }
    .sentiment-neutral  { color: #f59e0b; font-weight: 600; text-shadow: 0 0 5px rgba(245, 158, 11, 0.3); }

    /* File upload styling */
    .uploadedfile { background-color: #0a0a0a; border: 1px solid #444444; border-radius: 8px; color: #ffffff; }
    .stFileUploader > div { background-color: #0a0a0a; border: 2px dashed #444444; border-radius: 8px; }
    .stFileUploader > div:hover { border-color: #666666; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #000000; border-radius: 8px; gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        color: #888888;
        background-color: #0a0a0a;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #ffffff; background-color: #1a1a1a; }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        background-color: #1a1a1a;
        border-color: #2563eb;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.3);
    }

    /* DataFrame styling */
    .stDataFrame { background-color: #000000; }
    .stDataFrame [data-testid="stDataFrame"] { background-color: #0a0a0a; color: #ffffff; }

    /* Metrics styling */
    .metric-container { background-color: #0a0a0a; border: 1px solid #333333; border-radius: 8px; padding: 1rem; }
    .stMetric { background-color: #0a0a0a; }
    .stMetric > div { background-color: #0a0a0a; color: #ffffff; }

    /* Progress bar styling */
    .stProgress .st-bo { background-color: #333333; }
    .stProgress .st-bp { background-color: #2563eb; }

    /* Success/info/warning messages */
    .stSuccess { background-color: #0f3d2c; border: 1px solid #22c55e; color: #ffffff; }
    .stInfo    { background-color: #1e3a8a; border: 1px solid #3b82f6; color: #ffffff; }
    .stWarning { background-color: #451a03; border: 1px solid #f59e0b; color: #ffffff; }
    .stError   { background-color: #3d0c0c; border: 1px solid #888888; color: #ffffff; }

    /* Spinner styling */
    .stSpinner { color: #2563eb; }

    /* Column styling */
    .stColumn { background-color: transparent; }

    /* Links */
    a { color: #3b82f6; text-decoration: none; }
    a:hover { color: #60a5fa; text-decoration: underline; }

    /* Form submit buttons specifically */
    .stFormSubmitButton > button {
        background-color: #0A0A0A !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        margin-top: 1rem;
        min-width: 160px !important;
        width: 160px !important;
        padding: 0.6rem 1rem !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        background-color: #1a1a1a !important;
        border-color: #555555 !important;
        box-shadow: 0 2px 8px rgba(255, 255, 255, 0.1) !important;
    }

    /* Try Example button styling */
    .try-example-button > button {
        background-color: transparent !important;   /* remove fill */
        color: #ffffff !important;
        border: 1px solid #666666 !important;
        border-radius: 8px;
    }
    .try-example-button > button:hover {
        background-color: transparent !important;   /* stay transparent */
        border-color: #888888 !important;
        box-shadow: none !important;
    }

    /* Analyze Sentiment button styling */
    .analyze-button > button {
        background-color: #0A0A0A !important;
        color: #ffffff !important;
        border: none !important;                    /* remove stroke */
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    .analyze-button > button:hover {
        background-color: #1a1a1a !important;
        border: none !important;
        box-shadow: none !important;
    }

    .stApp .stTabs [data-baseweb="tab"],
    .stApp .stTabs [data-baseweb="tab"]:hover,
    .stApp .stTabs [aria-selected="true"],
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"] {
        border: none !important;
        box-shadow: none !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }
    
    .stApp .stTabs [data-baseweb="tab"]:focus,
    .stApp .stTabs [data-baseweb="tab"]:focus-visible,
    .stApp .stTabs [data-baseweb="tab"]:active,
    .stApp .stTabs [data-baseweb="tab"]:focus-within,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"]:focus,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"]:focus-visible,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"]:active {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }
    
    .stApp .stTabs [data-baseweb="tab"]::before,
    .stApp .stTabs [data-baseweb="tab"]::after,
    .stApp .stTabs [data-baseweb="tab-list"]::before,
    .stApp .stTabs [data-baseweb="tab-list"]::after,
    .stApp div[data-baseweb="tab-list"]::before,
    .stApp div[data-baseweb="tab-list"]::after,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"]::before,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"]::after {
        display: none !important;
        content: none !important;
        border: none !important;
        background: none !important;
        background-color: transparent !important;
    }
    
    .stApp .stTabs [data-baseweb="tab-list"]:focus,
    .stApp .stTabs [data-baseweb="tab-list"]:focus-within,
    .stApp div[data-baseweb="tab-list"]:focus,
    .stApp div[data-baseweb="tab-list"]:focus-within {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
        border-bottom: none !important;
    }
    
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"]:focus,
    .stApp .stTabs [data-baseweb="tab"][aria-selected="true"]:focus-visible,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"]:focus,
    .stApp div[data-baseweb="tab-list"] [data-baseweb="tab"][aria-selected="true"]:focus-visible {
        color: #2563eb !important;
        background-color: #1a1a1a !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
    }

    /* Placeholder text color */
    .stTextArea textarea::placeholder { color: #666666 !important; }

    /* Font settings */
    * {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif !important;
        font-size: 13px !important;
    }

    .stApp { text-align: left; }
    .block-container { text-align: left; max-width: 800px; margin: 0 auto; }

    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.3rem !important; }
    h4 { font-size: 1.1rem !important; }
    h5 { font-size: 1rem !important; }
    h6 { font-size: 0.9rem !important; }

    .main-title { font-size: 1.8rem !important; }
    .main-subtitle { font-size: 0.9rem !important; }

    .stButton > button { 
        font-size: 12px !important; 
        min-width: 160px !important;
        width: 160px !important;
        white-space: nowrap !important;
        text-align: center !important;
    }

    .stTextArea textarea { font-size: 12px !important; }
    .stTextInput input { font-size: 12px !important; }

    .stMetric .metric-value { font-size: 1.2rem !important; }
    .stMetric .metric-label { font-size: 0.8rem !important; }

    .stTabs [data-baseweb="tab"] { font-size: 12px !important; }

    p, span, div { color: #ffffff; }
    label { font-size: 11px !important; }

    .stContainer, .stColumn, .stColumns { text-align: left; }

    .main-header { text-align: center; }

    .stForm .stButton { text-align: center; margin: 0 8px; }

    .stFormSubmitButton { display: flex; justify-content: center; align-items: center; margin: 0 auto; }

    .stForm .stColumns > div { display: flex; justify-content: center; align-items: center; }

    .stApp > header { background-color: transparent; }
    .stApp [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333333; }

    /* Form labels */
    .stTextArea label { color: #ffffff !important; font-size: 11px !important; }
    .stFileUploader label { color: #ffffff !important; font-size: 11px !important; }

    /* Mode indicator badges */
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 10px !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 4px;
    }
    .mode-badge.strict { background-color: #1e3a8a; color: #60a5fa; border: 1px solid #3b82f6; }
    .mode-badge.lenient { background-color: #16a34a; color: #86efac; border: 1px solid #22c55e; }

    /* Clean Toggle Switch Styling */
    [data-testid="stToggle"] {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stToggle"] > div {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    [data-testid="stToggle"] label {
        color: #ffffff !important;
        font-size: 12px !important;
        order: 1 !important;
        margin-right: 8px !important;
    }
    
    /* Toggle switch container */
    [data-testid="stToggle"] [role="switch"] {
        order: 2 !important;
        background-color: #333333 !important;
        border: 2px solid #666666 !important;
        border-radius: 12px !important;
        width: 48px !important;
        height: 24px !important;
        position: relative !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
    }
    
    /* Toggle switch knob */
    [data-testid="stToggle"] [role="switch"] > div {
        background-color: #ffffff !important;
        border-radius: 50% !important;
        width: 18px !important;
        height: 18px !important;
        position: absolute !important;
        top: 1px !important;
        left: 1px !important;
        transition: transform 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        box-sizing: border-box !important;
    }
    
    /* Toggle switch checked state */
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
    }
    
    /* Toggle switch checked knob position - knob reaches right edge */
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] > div {
        background-color: #000000 !important;
        transform: translateX(24px) !important;
    }
    
    /* Toggle switch focus state */
    [data-testid="stToggle"] [role="switch"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3) !important;
    }
    .try-example-button > button {
    background-color: transparent !important;   /* remove fill */
    color: #ffffff !important;
    border: 1px solid #666666 !important;       /* keep subtle outline */
    box-shadow: none !important;
    }
    .try-example-button > button:hover,
    .try-example-button > button:focus {
    background-color: transparent !important;   /* stay transparent */
    border-color: #888888 !important;
    box-shadow: none !important;
    outline: none !important;
    }

    .analyze-button > button {
    background-color: #0A0A0A !important;
    color: #ffffff !important;
    border: none !important;                    /* no stroke */
    box-shadow: none !important;
    }
    .analyze-button > button:hover,
    .analyze-button > button:focus {
    background-color: #1a1a1a !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.6) !important;  /* white ring */
    border-color: #ffffff !important;                        /* white stroke when present */
    }

    .stButton > button:hover:focus {
    box-shadow: 0 0 0 2px rgba(255,255,255,0.6) !important;
    }

    
    html body .stApp [data-baseweb="tab"]:focus,
    html body .stApp [data-baseweb="tab"]:focus-visible,
    html body .stApp [data-baseweb="tab"]:hover,
    html body .stApp [data-baseweb="tab"]:active {
        border: none !important;
        box-shadow: none !important;
        background-color: #1a1a1a !important;
        outline: none !important;
    }
    
    html body .stApp *:focus,
    html body .stApp *:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.6) !important;
    }

    .stTextArea textarea:invalid,
    .stTextInput input:invalid {
        border-color: #888888 !important;
        box-shadow: none !important;
    }
    .stFileUploader [data-testid="stFileUploadDropzone"]:invalid,
    .stFileUploader [data-testid="stFileUploadDropzone"][aria-invalid="true"] {
        border-color: #888888 !important;
    }
    
    .stForm [data-testid="stFormSubmitButton"] button:focus,
    .stForm [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.6) !important;
        border-color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        border: none !important;
    }
    
    .stFileUploader > div[data-testid="stFileUploadDropzone"] {
        border-color: #444444 !important;
    }
    .stFileUploader > div[data-testid="stFileUploadDropzone"]:hover {
        border-color: #666666 !important;
    }
    .stFileUploader > div[data-testid="stFileUploadDropzone"]:focus,
    .stFileUploader > div[data-testid="stFileUploadDropzone"]:focus-within {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3) !important;
    }
    
    
    .stApp *:focus,
    .stApp *:focus-visible {
        outline: none !important;
    }
    
    [data-baseweb] *:focus,
    [data-baseweb] *:focus-visible,
    [data-baseweb] *:active {
        outline: none !important;
        border-color: #ffffff !important;
    }
    
    
    :root {
        --colors-primary: #ffffff !important;
        --colors-primary50: #ffffff !important;
        --colors-primary100: #ffffff !important;
        --colors-primary200: #ffffff !important;
        --colors-primary300: #ffffff !important;
        --colors-primary400: #ffffff !important;
        --colors-primary500: #ffffff !important;
        --colors-primary600: #ffffff !important;
        --colors-accent: #ffffff !important;
        --colors-accent50: #ffffff !important;
        --colors-accent100: #ffffff !important;
        --colors-accent200: #ffffff !important;
        --colors-accent300: #ffffff !important;
        --colors-accent400: #ffffff !important;
        --colors-accent500: #ffffff !important;
        --colors-negative: #888888 !important;
        --colors-negative50: #888888 !important;
        --colors-negative100: #888888 !important;
        --colors-negative200: #888888 !important;
        --colors-negative300: #888888 !important;
        --colors-negative400: #888888 !important;
        --colors-negative500: #888888 !important;
    }
    
    html body .stApp .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"],
    html body .stApp .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:focus,
    html body .stApp .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:focus-visible,
    html body .stApp .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:active {
        border-bottom-color: transparent !important;
        border-top-color: transparent !important;
        border-left-color: transparent !important;
        border-right-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid transparent !important;
        border-color: transparent !important;
        background-color: #1a1a1a !important;
        color: #2563eb !important;
    }
    [data-baseweb="tab"][aria-selected="true"]:focus,
    [data-baseweb="tab"][aria-selected="true"]:focus-visible {
        border-bottom: 2px solid transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    
    [data-baseweb*="tab"]:focus {
        border-bottom: none !important;
        box-shadow: none !important;
    }
    
    /* Reduce spacing between mode selector and form elements */
    .stTabs [data-baseweb="tab-panel"] > div > div:first-child + .stForm {
        margin-top: -1rem !important;
    }
    
    /* Reduce spacing between mode selector and file uploader */
    .stTabs [data-baseweb="tab-panel"] > div > div:first-child + div .stFileUploader {
        margin-top: -1rem !important;
    }
    
    /* Alternative approach - target any element following the mode selector columns */
    .stTabs [data-baseweb="tab-panel"] > div > div:first-child + * {
        margin-top: -0.75rem !important;
    }

</style>

<script>
// Simple JavaScript to add button styling classes
function applyButtonStyles() {
    document.querySelectorAll('button').forEach(btn => {
        if (btn.innerText.includes('Try Example')) {
            btn.parentElement.classList.add('try-example-button');
        }
        if (btn.innerText.includes('Analyze Sentiment')) {
            btn.parentElement.classList.add('analyze-button');
        }
    });
}

// Apply button styles when page loads
setTimeout(applyButtonStyles, 100);
setTimeout(applyButtonStyles, 500);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyButtonStyles);
} else {
    applyButtonStyles();
}

// Simple observer to reapply button styles when buttons are added
const observer = new MutationObserver(() => {
    applyButtonStyles();
});

observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

def display_sentiment_result(result, analysis_mode=None):
    """Show the analysis results in a nice, formatted way."""
    # Handle case where analysis failed
    if not result or 'label' not in result:
        st.error("Unable to analyze sentiment. Please try again.")
        return

    # Extract the key results from the analysis
    label = result.get('label', 'Neutral')
    confidence = result.get('confidence', 0.0)
    explanation = result.get('explanation', 'No explanation available')
    evidence = result.get('evidence_phrases', [])

    # Pick the right color styling based on sentiment
    sentiment_display = {'Positive': 'sentiment-positive', 'Negative': 'sentiment-negative', 'Neutral': 'sentiment-neutral'}
    css_class = sentiment_display.get(label, 'sentiment-neutral')

    # Add a badge to show which analysis mode was used
    mode_badge = ""
    if analysis_mode:
        badge_class = "strict" if analysis_mode == "strict" else "lenient"
        mode_badge = f'<span class="mode-badge {badge_class}">{analysis_mode.title()} Mode</span>'

    st.markdown(f"""
    <div class="result-container">
        <h3><span class="{css_class}">{label}</span> {mode_badge}</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Analysis:</strong> {explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    # Show the specific phrases that led to this decision
    if evidence:
        st.write("**Key phrases that influenced this classification:**")
        for phrase in evidence[:5]:
            st.write(f"• {phrase}")


def render_analysis_mode_selector(key_suffix=""):
    """Create the toggle switch for choosing strict vs normal analysis."""
    # Use Streamlit's native column system for reliable positioning
    if key_suffix in ["_single", "_batch"]:
        text_content = """<span style="color:#ffffff; font-weight:bold;">STRICT MODE:</span> <span style="color:#b0b0b0;">Conservative analysis defaults ambiguous text to Neutral, while Normal mode detects subtler sentiment cues.</span>"""
        font_size = "8px"
    else:
        text_content = """<span style="color:#ffffff; font-weight:bold;">Strict Mode:</span> <span style="color:#b0b0b0;">Only flags clear, strong sentiment and defaults borderline text to Neutral, while Normal mode accepts subtler cues as Positive/Negative.</span>"""
        font_size = "10px"
    
    # Use columns with extreme right positioning - much wider left column to push toggle to extreme right
    col1, col2 = st.columns([16, 1], vertical_alignment="center")
    
    with col1:
        # Wrap text in a flex container for vertical centering
        st.markdown(f"""
        <div style="display: flex; align-items: center; height: 100%; min-height: 24px;">
            <p style="color:#b0b0b0; font-size:{font_size} !important; line-height:1.3; margin:0; padding:0; word-wrap: break-word; overflow-wrap: break-word; white-space: normal; overflow: visible; flex: 1;">
                {text_content}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Container positioned at EXTREME RIGHT edge
        st.markdown("""
        <div style="
            width: 100%; 
            display: flex; 
            align-items: center; 
            height: 100%; 
            min-height: 24px; 
            justify-content: flex-end;
            margin: 0;
            padding: 0;
        ">
        """, unsafe_allow_html=True)
        
        # No additional margin - toggle should be flush right
        strict_mode = st.toggle(
            "Strict Mode",
            value=False,
            key=f"strict_mode_toggle{key_suffix}",
            help="Strict: Only flags clear sentiment. Normal: Accepts subtle cues.",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    return "strict" if strict_mode else "lenient"


    

def create_download_link(df, filename="sentiment_results.csv"):
    """Create a download link so users can save their batch results."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def main():
    """The main app - handles both single review analysis and batch processing."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 Movie Review Sentiment Analyzer</h1>
        <p class="main-subtitle">Discover the emotional tone of movie reviews using sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two main tabs for different use cases
    tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

    # Tab 1: Analyze one review at a time
    with tab1:
        analysis_mode = render_analysis_mode_selector("_single")
        
        with st.form("review_form"):
            review_text = st.text_area(
                "Movie Review",
                height=120,
                placeholder="Enter your movie review here... For example: \"This movie was absolutely incredible! The cinematography was stunning...\"",
                label_visibility="collapsed"
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.empty()
            with col2:
                button_col1, button_col2 = st.columns([1, 1])
                with button_col1:
                    try_example = st.form_submit_button("Try Example", type="secondary")
                with button_col2:
                    analyze_button = st.form_submit_button("Analyze Sentiment", type="primary")
            with col3:
                st.empty()

        # Handle the "Try Example" button - picks a random sample review
        if try_example:
            example_reviews = [
                "This movie was absolutely fantastic! Amazing acting and incredible plot twists.",
                "Terrible movie. Poor acting, boring plot, and a complete waste of time. I walked out halfway through.",
                "The movie was okay. Not bad but not great either. Some good moments but overall pretty average.",
                "The film has some interesting visual elements and decent performances, though the plot feels a bit rushed.",
                "I loved the cinematography but found the dialogue somewhat predictable. Overall entertaining."
            ]
            import random
            selected_example = random.choice(example_reviews)
            st.text_area("Example review:", value=selected_example, height=80, disabled=True)
            with st.spinner(f"Analyzing example review in {analysis_mode} mode..."):
                result = analyze_sentiment(selected_example, analysis_mode=analysis_mode)
                display_sentiment_result(result, analysis_mode)

        # Handle the main "Analyze Sentiment" button
        if analyze_button and review_text.strip():
            with st.spinner(f"Analyzing sentiment in {analysis_mode} mode..."):
                result = analyze_sentiment(review_text.strip(), analysis_mode=analysis_mode)
                display_sentiment_result(result, analysis_mode)
        elif analyze_button and not review_text.strip():
            st.warning("Please enter a review to analyze.")

    # Tab 2: Process multiple reviews from a CSV file
    with tab2:
        analysis_mode_batch = render_analysis_mode_selector("_batch")
        
        # File uploader for batch processing
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should have a 'review' column containing the movie reviews to analyze"
        )

        # Process the uploaded CSV file
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Make sure the file has the expected format
                if 'review' not in df.columns:
                    st.error("CSV file must contain a 'review' column with the movie reviews.")
                    st.info("Expected format: CSV with at least a 'review' column containing text to analyze.")
                    return

                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head(), use_container_width=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"Found {len(df)} reviews to analyze")
                with col2:
                    process_batch = st.button("Process All Reviews", type="primary")

                # Handle the "Process All Reviews" button
                if process_batch:
                    st.info(f"Processing {len(df)} reviews using **{analysis_mode_batch.title()} Mode** analysis...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process each review one by one with progress tracking
                    results = []
                    total_reviews = len(df)
                    for i, row in df.iterrows():
                        progress = (i + 1) / total_reviews
                        progress_bar.progress(progress)
                        status_text.text(f"Processing review {i + 1} of {total_reviews} ({analysis_mode_batch} mode)")

                        # Handle empty or invalid reviews gracefully
                        review = str(row['review']).strip()
                        if not review or review.lower() in ['nan', 'none', '']:
                            result = {
                                'predicted_sentiment': 'Neutral',
                                'confidence': 0.0,
                                'explanation': 'Empty or invalid review',
                                'evidence_phrases': '',
                                'analysis_mode': analysis_mode_batch
                            }
                        # Analyze the review and handle any errors
                        else:
                            try:
                                analysis = analyze_sentiment(review, analysis_mode=analysis_mode_batch)
                                result = {
                                    'predicted_sentiment': analysis['label'],
                                    'confidence': analysis['confidence'],
                                    'explanation': analysis['explanation'],
                                    'evidence_phrases': ', '.join(analysis.get('evidence_phrases', [])),
                                    'analysis_mode': analysis_mode_batch
                                }
                            except Exception as e:
                                # If analysis fails, provide a safe default
                                result = {
                                    'predicted_sentiment': 'Neutral',
                                    'confidence': 0.0,
                                    'explanation': f'Analysis failed: {str(e)}',
                                    'evidence_phrases': '',
                                    'analysis_mode': analysis_mode_batch
                                }

                        results.append(result)
                        time.sleep(0.1)  # Small delay to avoid overwhelming the API

                    # Combine original data with analysis results
                    results_df = df.copy()
                    if results:  # Make sure we have results before processing
                        # Get the keys from the first result dictionary
                        for key in results[0].keys():
                            results_df[key] = [r[key] for r in results]

                    # Clean up progress indicators and show success
                    progress_bar.empty()
                    status_text.empty()

                    st.success(f"Successfully analyzed {len(df)} reviews using **{analysis_mode_batch.title()} Mode**!")

                    # Show summary statistics in a nice layout
                    sentiment_counts = results_df['predicted_sentiment'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = sentiment_counts.get('Positive', 0)
                        st.metric("Positive", positive_count, f"{positive_count/len(df)*100:.1f}%")
                    with col2:
                        negative_count = sentiment_counts.get('Negative', 0)
                        st.metric("Negative", negative_count, f"{negative_count/len(df)*100:.1f}%")
                    with col3:
                        neutral_count = sentiment_counts.get('Neutral', 0)
                        st.metric("Neutral", neutral_count, f"{neutral_count/len(df)*100:.1f}%")

                    # Show overall confidence and preview of results
                    avg_confidence = results_df['confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")

                    st.write("**Results Preview:**")
                    st.dataframe(results_df.head(10), use_container_width=True)

                    # Provide download link for the complete results
                    st.markdown("**Download Complete Results:**")
                    csv_download = create_download_link(results_df, "movie_sentiment_analysis_results.csv")
                    st.markdown(csv_download, unsafe_allow_html=True)

            # Handle file processing errors
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please make sure your CSV file is properly formatted with a 'review' column.")

if __name__ == "__main__":
    main()
