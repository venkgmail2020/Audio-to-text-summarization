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
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .keyword-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎤 Audio to Text Summarizer</h1></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### Configuration")
    
    with st.expander("API Settings", expanded=True):
        assembly_key = st.text_input(
            "AssemblyAI Key",
            value=st.session_state.assembly_key,
            type="password"
        )
        if st.button("Save", use_container_width=True):
            st.session_state.assembly_key = assembly_key
            st.success("✅ Saved!")
    
    st.markdown("### Quick Stats")
    if st.session_state.current_summary:
        words = len(st.session_state.current_summary.split())
        st.metric("Last Summary Words", words)
    
    st.markdown("### Supported Formats")
    st.markdown("🎥 **Video:** MP4, AVI, MOV")
    st.markdown("🎵 **Audio:** MP3, WAV, M4A")
    st.markdown("📄 **Document:** PDF, TXT")
    st.markdown("🌐 **Online:** URLs, YouTube")

# ===== FUNCTIONS =====
def extract_pdf_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

def generate_summary(text, num_sentences=3):
    try:
        if not text or len(text) < 100:
            return text
        
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join(str(s) for s in summary)
    except:
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:num_sentences])

def get_keywords(text):
    try:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stopwords = set(nltk.corpus.stopwords.words('english'))
        filtered = [w for w in words if w not in stopwords]
        return Counter(filtered).most_common(8)
    except:
        return []

def text_to_speech(text):
    try:
        tts = gTTS(text=text[:500], lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except:
        return None

# ===== MAIN TABS =====
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🔗 URL/YouTube", "📝 Paste Text"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov']
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext in ['mp4', 'avi', 'mov']:
            st.video(uploaded_file)
        elif file_ext in ['mp3', 'wav']:
            st.audio(uploaded_file)
        
        if st.button("Process", key="process_file"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                if file_ext == 'pdf':
                    text = extract_pdf_text(path)
                elif file_ext == 'txt':
                    with open(path, 'r') as f:
                        text = f.read()
                else:
                    # Simulate transcription for demo
                    time.sleep(2)
                    text = f"Sample transcription from {uploaded_file.name}"
                
                os.unlink(path)
                
                if text:
                    st.session_state.current_text = text
                    st.session_state.show_results = True
                    st.rerun()

with tab2:
    url = st.text_input("Enter URL", placeholder="https://youtube.com/...")
    if url and st.button("Fetch"):
        with st.spinner("Fetching..."):
            time.sleep(1)
            st.session_state.current_text = f"Sample content from {url}"
            st.session_state.show_results = True
            st.rerun()

with tab3:
    text_input = st.text_area("Paste text", height=150)
    if text_input and st.button("Summarize"):
        if len(text_input) > 50:
            st.session_state.current_text = text_input
            st.session_state.show_results = True
            st.rerun()
        else:
            st.warning("Text too short")

# ===== RESULTS SECTION =====
if st.session_state.show_results and st.session_state.current_text:
    st.markdown("---")
    
    text = st.session_state.current_text
    summary = generate_summary(text)
    st.session_state.current_summary = summary
    keywords = get_keywords(text)
    
    # Original text preview
    with st.expander("📄 Full Text", expanded=False):
        st.write(text[:500] + "..." if len(text) > 500 else text)
    
    # Summary
    st.subheader("📝 Summary")
    st.info(summary)
    
    # Stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    words_count = len(text.split())
    sentences_count = len(nltk.sent_tokenize(text))
    summary_words = len(summary.split())
    
    with col1:
        st.markdown(f"<div class='stat-card'><b>Words</b><br><h2>{words_count}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><b>Sentences</b><br><h2>{sentences_count}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-card'><b>Summary Words</b><br><h2>{summary_words}</h2></div>", unsafe_allow_html=True)
    with col4:
        read_time = f"{words_count//200} min"
        st.markdown(f"<div class='stat-card'><b>Read Time</b><br><h2>{read_time}</h2></div>", unsafe_allow_html=True)
    
    # Keywords
    if keywords:
        st.subheader("🏷️ Keywords")
        html = ""
        for word, count in keywords:
            html += f"<span class='keyword-tag'>{word}</span> "
        st.markdown(html, unsafe_allow_html=True)
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📄 Download Text", text, "full_text.txt")
    with col2:
        st.download_button("📝 Download Summary", summary, "summary.txt")
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio)
            st.download_button("🔊 Download Audio", audio, "summary.mp3")
    
    # Clear button
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.current_text = ''
        st.session_state.current_summary = ''
        st.rerun()
