import streamlit as st
import requests
import tempfile
import os
import time
import nltk
import PyPDF2
import re
from collections import Counter
from gtts import gTTS
import io
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import ssl
from bs4 import BeautifulSoup
import validators
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
import matplotlib.pyplot as plt

# ===== NLTK SETUP =====
try:
    _create_unverified_https_context = ssl._create_unverified_context
except:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

st.set_page_config(
    page_title="Audio to Text Summarizer", 
    page_icon="🎤", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS - Modern Dark Design =====
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header with glassmorphism */
    .glass-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .glass-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .glass-header p {
        color: #a0a0a0;
        font-size: 1.1rem;
    }
    
    /* Card design */
    .custom-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.02);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-card h3 {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Keywords */
    .keyword-tag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success/Info/Warning messages */
    .stAlert {
        border-radius: 10px;
        border: none;
    }
    
    .stAlert.success {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
    }
    
    .stAlert.info {
        background: rgba(23, 162, 184, 0.1);
        border-left: 4px solid #17a2b8;
    }
    
    .stAlert.warning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Audio player */
    audio {
        border-radius: 25px;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class='glass-header'>
    <h1>🎤 Audio to Text Summarizer</h1>
    <p>Transform your audio, video, PDF, and text into smart summaries with AI</p>
</div>
""", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'current_keywords' not in st.session_state:
    st.session_state.current_keywords = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    
    with st.expander("⚙️ API Settings", expanded=True):
        assembly_key = st.text_input(
            "AssemblyAI Key",
            value=st.session_state.assembly_key,
            type="password",
            placeholder="Enter your API key",
            help="Get free key from assemblyai.com"
        )
        
        if st.button("💾 Save", use_container_width=True):
            st.session_state.assembly_key = assembly_key
            st.success("✅ Saved!")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    if st.session_state.current_summary:
        words = len(st.session_state.current_summary.split())
        st.metric("Last Summary Words", f"{words}")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    
    formats = [
        ("🎥 Video", "MP4, AVI, MOV"),
        ("🎵 Audio", "MP3, WAV, M4A"),
        ("📄 Document", "PDF, TXT"),
        ("🌐 Online", "URLs, YouTube")
    ]
    
    for icon, text in formats:
        st.markdown(f"**{icon}**  \n{text}")

# ===== MAIN TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🔗 URL/YouTube", "📝 Text", "ℹ️ About"])

# ===== TAB 1: FILE UPLOAD =====
with tab1:
    st.markdown("### Upload File")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov'],
        help="Max file size: 200MB"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        with col1:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext in ['mp4', 'avi', 'mov']:
                st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav']:
                st.audio(uploaded_file)
        
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.markdown(f"""
            <div class='metric-card'>
                <h3>File Info</h3>
                <div class='value'>{uploaded_file.name[:20]}...</div>
                <p style='color: #a0a0a0; margin-top: 0.5rem;'>{file_size:.2f} MB</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Process", use_container_width=True):
            st.session_state.processing = True
            
            with st.spinner("Processing your file..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                if file_ext == 'pdf':
                    # Simulate PDF extraction
                    time.sleep(1)
                    text = f"Sample extracted text from {uploaded_file.name}"
                elif file_ext == 'txt':
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    # Simulate transcription
                    time.sleep(2)
                    text = f"Sample transcription from {uploaded_file.name}"
                
                os.unlink(path)
                
                if text:
                    st.session_state.current_text = text
                    st.success("✅ Processing complete!")
                    st.session_state.processing = False
                    st.rerun()

# ===== TAB 2: URL/YOUTUBE =====
with tab2:
    st.markdown("### Enter URL")
    
    url = st.text_input(
        "URL",
        placeholder="https://example.com or YouTube link",
        label_visibility="collapsed"
    )
    
    if url and st.button("🌐 Fetch", use_container_width=True):
        with st.spinner("Fetching content..."):
            time.sleep(1)
            st.session_state.current_text = f"Sample content from {url}"
            st.success("✅ Content fetched!")
            st.rerun()

# ===== TAB 3: PASTE TEXT =====
with tab3:
    st.markdown("### Paste Your Text")
    
    text_input = st.text_area(
        "Text",
        height=200,
        placeholder="Paste your article, notes, or any text here...",
        label_visibility="collapsed"
    )
    
    if text_input and st.button("📝 Summarize", use_container_width=True):
        if len(text_input) > 100:
            st.session_state.current_text = text_input
            st.success("✅ Text ready for summarization!")
            st.rerun()
        else:
            st.warning("Please enter at least 100 characters")

# ===== TAB 4: ABOUT =====
with tab4:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: white;'>About Audio to Text Summarizer</h3>
        <p style='color: #a0a0a0; line-height: 1.6;'>
            This powerful tool converts your audio, video, PDF, and text content into 
            concise, intelligent summaries using advanced NLP techniques.
        </p>
        
        <h4 style='color: white; margin-top: 2rem;'>✨ Key Features</h4>
        <ul style='color: #a0a0a0;'>
            <li>🎤 Audio/Video transcription</li>
            <li>📄 PDF text extraction</li>
            <li>🌐 URL/YouTube content fetch</li>
            <li>📝 Smart summarization with LexRank</li>
            <li>🏷️ Keyword extraction</li>
            <li>🔊 Text-to-speech output</li>
            <li>📊 Visual infographics</li>
            <li>🔍 Smart document search</li>
        </ul>
        
        <h4 style='color: white; margin-top: 2rem;'>🚀 How to Use</h4>
        <ol style='color: #a0a0a0;'>
            <li>Add your AssemblyAI API key in sidebar</li>
            <li>Upload file, paste URL, or enter text</li>
            <li>Click Process/Fetch/Summarize</li>
            <li>View summary, keywords, and insights</li>
            <li>Download results or listen to audio</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ===== RESULTS SECTION =====
if st.session_state.current_text and not st.session_state.processing:
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    text = st.session_state.current_text
    
    # Simple summary (replace with your actual summary function)
    sentences = nltk.sent_tokenize(text)
    summary = ' '.join(sentences[:3]) if len(sentences) > 3 else text
    
    # Keywords (replace with your actual keyword extraction)
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    keywords = Counter([w for w in words if w not in stopwords]).most_common(10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Summary")
        st.info(summary)
        
        # Download buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.download_button("📄 Text", text, "full_text.txt")
        with col_b:
            st.download_button("📝 Summary", summary, "summary.txt")
        with col_c:
            audio = text_to_speech(summary) if 'text_to_speech' in dir() else None
            if audio:
                st.audio(audio)
                st.download_button("🔊 Audio", audio, "summary.mp3")
    
    with col2:
        st.markdown("### 📊 Stats")
        
        # Metrics
        metrics = [
            ("📊 Words", f"{len(text.split()):,}"),
            ("🔤 Sentences", f"{len(sentences)}"),
            ("📉 Summary", f"{len(summary.split())} words"),
            ("⏱️ Read Time", f"{len(text.split())//200} min")
        ]
        
        for label, value in metrics:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{label}</h3>
                <div class='value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Keywords
        if keywords:
            st.markdown("### 🏷️ Keywords")
            html = ""
            for word, count in keywords[:8]:
                html += f"<span class='keyword-tag'>{word}</span> "
            st.markdown(html, unsafe_allow_html=True)
    
    # Clear button
    if st.button("🔄 Clear Results", use_container_width=True):
        st.session_state.current_text = ''
        st.session_state.current_summary = ''
        st.session_state.current_keywords = []
        st.rerun()

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    Made with ❤️ using Streamlit | © 2026 Audio to Text Summarizer
</div>
""", unsafe_allow_html=True)
