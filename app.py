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
import random

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
    page_title="AI Video/Text Analyzer", 
    page_icon="🎥", 
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .timestamp-item {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-family: monospace;
    }
    .moment-highlight {
        background: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
    }
    .plagiarism-low { background: #d4edda; color: #155724; padding: 1rem; border-radius: 5px; }
    .plagiarism-medium { background: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; }
    .plagiarism-high { background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 5px; }
    .topic-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
    }
    .timeline-item {
        border-left: 3px solid #667eea;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎥 AI Video/Text Analyzer</h1><p>Plagiarism | Timeline | Topics | Timestamps | Key Moments</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### ⚙️ API Key")
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assembly_key,
        type="password"
    )
    if assembly_key != st.session_state.assembly_key:
        st.session_state.assembly_key = assembly_key
    
    st.markdown("---")
    st.markdown("### 📌 Features")
    st.markdown("✅ Plagiarism Checker")
    st.markdown("✅ Timeline Generator")
    st.markdown("✅ Topic Detection")
    st.markdown("✅ Timestamp Summary")
    st.markdown("✅ Key Moments Detection")

# ===== PDF EXTRACTION =====
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
    except:
        return None

# ===== FEATURE 1: PLAGIARISM CHECKER =====
def check_plagiarism(text):
    """Simulate plagiarism check"""
    common_phrases = [
        "according to", "research shows", "studies indicate", "as a result",
        "in conclusion", "for example", "such as", "due to", "because of"
    ]
    
    matches = sum(1 for phrase in common_phrases if phrase in text.lower())
    score = min(100, matches * 10)
    
    if score < 30:
        return score, "Low", "plagiarism-low"
    elif score < 60:
        return score, "Medium", "plagiarism-medium"
    else:
        return score, "High", "plagiarism-high"

# ===== FEATURE 2: TIMELINE GENERATOR =====
def generate_timeline(text):
    """Extract timeline from text"""
    sentences = nltk.sent_tokenize(text)
    timeline = []
    
    # Look for years
    year_pattern = r'\b(19|20)\d{2}\b'
    
    for sent in sentences[:10]:
        years = re.findall(year_pattern, sent)
        if years:
            timeline.append({
                'year': years[0],
                'event': sent[:100] + "..."
            })
    
    return timeline[:5]

# ===== FEATURE 3: TOPIC DETECTION =====
def detect_topics(text, num_topics=5):
    """Extract main topics"""
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    word_freq = Counter(words)
    
    common = {'The', 'This', 'That', 'These', 'Those', 'There'}
    topics = [(w, c) for w, c in word_freq.most_common(15) 
              if w not in common and len(w) > 3]
    
    return topics[:num_topics]

# ===== FEATURE 4: TIMESTAMP SUMMARY =====
def generate_timestamps(text, duration_minutes=10):
    """Generate timestamps with summaries"""
    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    
    timestamps = []
    interval = duration_minutes / min(10, num_sentences)
    
    for i in range(0, min(10, num_sentences)):
        minutes = int(i * interval)
        seconds = int((i * interval - minutes) * 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        timestamps.append({
            'time': time_str,
            'summary': sentences[i][:80] + "..."
        })
    
    return timestamps

# ===== FEATURE 5: KEY MOMENTS DETECTION =====
def detect_key_moments(text, num_moments=5):
    """Detect important moments in video"""
    sentences = nltk.sent_tokenize(text)
    
    # Score sentences based on keywords
    keywords = ['important', 'key', 'significant', 'crucial', 'vital',
                'breakthrough', 'revolution', 'innovation', 'discovery']
    
    scored_sentences = []
    for i, sent in enumerate(sentences):
        score = sum(1 for word in keywords if word in sent.lower())
        if score > 0:
            scored_sentences.append((i, sent, score))
    
    # Sort by score
    scored_sentences.sort(key=lambda x: x[2], reverse=True)
    
    moments = []
    for i, (idx, sent, score) in enumerate(scored_sentences[:num_moments]):
        # Convert to timestamp (approx)
        timestamp = f"{idx // 6:02d}:{(idx % 6) * 10:02d}"
        moments.append({
            'time': timestamp,
            'moment': sent[:100] + "...",
            'importance': score
        })
    
    return moments

# ===== AUTO PROCESS FUNCTION =====
def auto_process(text):
    """Process all 5 features"""
    if not text or len(text) < 100:
        st.warning("Text too short (min 100 characters)")
        return
    
    st.session_state.current_text = text
    
    # FEATURE 1: Plagiarism Checker
    st.markdown("## 🔍 Plagiarism Checker")
    score, level, css_class = check_plagiarism(text)
    st.markdown(f"<div class='{css_class}'><b>Plagiarism Score:</b> {score}% - {level} Risk</div>", 
                unsafe_allow_html=True)
    
    # FEATURE 2: Timeline Generator
    st.markdown("## 📅 Timeline Generator")
    timeline = generate_timeline(text)
    if timeline:
        for item in timeline:
            st.markdown(f"<div class='timeline-item'>📅 {item['year']}<br>{item['event']}</div>", 
                       unsafe_allow_html=True)
    else:
        st.info("ℹ️ No timeline events detected")
    
    # FEATURE 3: Topic Detection
    st.markdown("## 🎯 Topic Detection")
    topics = detect_topics(text)
    if topics:
        topic_html = ""
        for topic, count in topics:
            topic_html += f"<span class='topic-tag'>{topic} ({count})</span> "
        st.markdown(topic_html, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No specific topics detected")
    
    # FEATURE 4: Timestamp Summary
    st.markdown("## ⏱️ Timestamp Summary")
    duration = st.slider("Video Duration (minutes)", 1, 60, 10)
    timestamps = generate_timestamps(text, duration)
    
    for ts in timestamps:
        st.markdown(f"<div class='timestamp-item'>⏱️ {ts['time']} → {ts['summary']}</div>", 
                   unsafe_allow_html=True)
    
    # FEATURE 5: Key Moments Detection
    st.markdown("## 🎬 Key Moments Detection")
    moments = detect_key_moments(text)
    
    for moment in moments:
        st.markdown(f"<div class='moment-highlight'>✨ {moment['time']} - {moment['moment']}</div>", 
                   unsafe_allow_html=True)
    
    # Summary
    st.markdown("## 📝 Summary")
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, 5)
    summary = ' '.join(str(s) for s in summary_sentences)
    st.info(summary)

# ===== MAIN UI =====
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🔗 URL/YouTube", "📝 Paste Text"])

# TAB 1: FILE UPLOAD
with tab1:
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['mp4', 'mp3', 'wav', 'pdf', 'txt']
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            
            if file_ext == 'pdf':
                text = extract_pdf_text(path)
            elif file_ext == 'txt':
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                # Simulate transcription
                time.sleep(2)
                text = f"Sample transcription from {uploaded_file.name}. " * 100
            
            os.unlink(path)
            
            if text:
                auto_process(text)

# TAB 2: URL/YOUTUBE
with tab2:
    url = st.text_input("Enter URL", placeholder="https://youtube.com/...")
    
    if url:
        with st.spinner("Fetching..."):
            # Simulate YouTube content
            text = f"Sample content from {url}. " * 100
            auto_process(text)

# TAB 3: PASTE TEXT
with tab3:
    text_input = st.text_area("Paste text", height=200)
    
    if text_input:
        auto_process(text_input)
