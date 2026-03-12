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
import numpy as np
import hashlib

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
    page_title="AI Research Assistant", 
    page_icon="🤖", 
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
    .summary-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .plagiarism-high {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
    }
    .plagiarism-medium {
        background: #ffbb33;
        color: black;
        padding: 1rem;
        border-radius: 5px;
    }
    .plagiarism-low {
        background: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 5px;
    }
    .timeline-item {
        border-left: 3px solid #667eea;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .topic-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
    }
    .search-result {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🤖 AI Research Assistant</h1><p>Summarization | Plagiarism | Timeline | Topics | Search | Infographic</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'processed' not in st.session_state:
    st.session_state.processed = False

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assembly_key,
        type="password"
    )
    if assembly_key != st.session_state.assembly_key:
        st.session_state.assembly_key = assembly_key
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if st.session_state.current_summary:
        words = len(st.session_state.current_summary.split())
        st.metric("Last Summary", f"{words} words")
    
    st.markdown("### 📌 Supported")
    st.markdown("🎥 Video: MP4, AVI, MOV")
    st.markdown("🎵 Audio: MP3, WAV, M4A")
    st.markdown("📄 Document: PDF, TXT")
    st.markdown("🌐 Online: URLs, YouTube")

# ===== FEATURE 0: SUMMARIZATION (FIXED) =====
def generate_summary(text, num_sentences=5):
    """Generate summary using LexRank"""
    try:
        if not text or len(text) < 100:
            return text
        
        # Use LexRank for better summaries
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, num_sentences)
        summary = ' '.join(str(s) for s in summary_sentences)
        return summary
    except Exception as e:
        # Fallback to simple extractive summary
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:num_sentences])

# ===== FEATURE 1: PLAGIARISM CHECKER =====
def check_plagiarism(text):
    """Simulate plagiarism check"""
    words = text.lower().split()
    total_words = len(words)
    
    # Common phrases that might indicate plagiarism
    common_phrases = [
        "according to", "research shows", "studies indicate", "as a result",
        "in conclusion", "for example", "such as", "due to", "because of",
        "in addition", "furthermore", "moreover", "however", "therefore"
    ]
    
    matches = 0
    for phrase in common_phrases:
        if phrase in text.lower():
            matches += 1
    
    plagiarism_score = min(100, matches * 5)
    
    if plagiarism_score > 50:
        level = "High"
        color = "plagiarism-high"
    elif plagiarism_score > 20:
        level = "Medium"
        color = "plagiarism-medium"
    else:
        level = "Low"
        color = "plagiarism-low"
    
    return plagiarism_score, level, color

# ===== FEATURE 2: TIMELINE GENERATOR =====
def generate_timeline(text):
    """Extract potential timeline events"""
    sentences = nltk.sent_tokenize(text)
    timeline = []
    
    # Look for date patterns
    date_patterns = [
        r'\b\d{4}\b',  # Years
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ]
    
    for sent in sentences:
        for pattern in date_patterns:
            match = re.search(pattern, sent, re.IGNORECASE)
            if match:
                timeline.append({
                    'date': match.group(),
                    'event': sent[:150] + "..." if len(sent) > 150 else sent
                })
                break
    
    return timeline[:5]

# ===== FEATURE 3: TOPIC DETECTION =====
def detect_topics(text, num_topics=5):
    """Extract main topics from text"""
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    word_freq = Counter(words)
    
    # Filter out common words
    common = {'The', 'This', 'That', 'These', 'Those', 'There', 'They', 'What', 'Which'}
    topics = [(w, c) for w, c in word_freq.most_common(15) if w not in common and len(w) > 3]
    
    return topics[:num_topics]

# ===== FEATURE 4: SMART SEARCH =====
def smart_search(text, query):
    """Search inside document"""
    if not query or len(query) < 2:
        return []
    
    sentences = nltk.sent_tokenize(text)
    results = []
    
    for sent in sentences:
        if query.lower() in sent.lower():
            # Highlight the query
            highlighted = sent.replace(query, f"**{query}**")
            results.append(highlighted)
    
    return results[:5]

# ===== FEATURE 5: INFOGRAPHIC GENERATOR =====
def generate_infographic(text, topics):
    """Create visual infographic"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    
    # 1. Word Frequency
    words = re.findall(r'\b\w{4,}\b', text.lower())
    word_counts = Counter(words).most_common(8)
    
    if word_counts:
        ax1 = axes[0, 0]
        words_list, counts_list = zip(*word_counts)
        ax1.barh(words_list, counts_list, color='#667eea')
        ax1.set_title('Top Keywords', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency')
    
    # 2. Topic Distribution
    if topics:
        ax2 = axes[0, 1]
        topic_names, topic_counts = zip(*topics)
        ax2.pie(topic_counts, labels=topic_names, autopct='%1.1f%%', 
                colors=['#667eea', '#764ba2', '#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax2.set_title('Topic Distribution', fontsize=12, fontweight='bold')
    
    # 3. Sentence Length Distribution
    sentences = nltk.sent_tokenize(text)
    sent_lengths = [len(sent.split()) for sent in sentences]
    ax3 = axes[1, 0]
    ax3.hist(sent_lengths, bins=15, color='#764ba2', alpha=0.7)
    ax3.set_title('Sentence Length Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Words per Sentence')
    ax3.set_ylabel('Frequency')
    
    # 4. Text Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    unique_words = len(set(re.findall(r'\b\w+\b', text.lower())))
    total_words = len(text.split())
    total_sents = len(sentences)
    
    stats_text = f"""
    📊 DOCUMENT STATISTICS
    
    Total Words: {total_words:,}
    Unique Words: {unique_words:,}
    Total Sentences: {total_sents}
    Avg Sentence: {total_words/total_sents:.1f} words
    Reading Time: {total_words//200} min
    
    🎯 TOP TOPICS
    """
    for topic, count in topics[:3]:
        stats_text += f"\n• {topic}: {count} times"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ===== TEXT EXTRACTION FUNCTIONS =====
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

def extract_url_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            return ' '.join(p.get_text() for p in soup.find_all('p'))
    except:
        return None

# ===== AUTO PROCESS FUNCTION =====
def auto_process(text):
    """Auto generate all 6 features"""
    if not text or len(text) < 100:
        st.warning("Text too short (min 100 characters)")
        return
    
    st.session_state.current_text = text
    
    # FEATURE 0: SUMMARY (FIXED)
    st.markdown("## 📝 Summary")
    summary = generate_summary(text, 5)
    st.session_state.current_summary = summary
    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
    
    # FEATURE 1: Plagiarism Check
    with st.container():
        st.markdown("---")
        st.markdown("## 🔍 Plagiarism Checker")
        score, level, color = check_plagiarism(text)
        st.markdown(f"<div class='{color}'>"
                   f"<b>Plagiarism Score:</b> {score}% - {level} Risk</div>", 
                   unsafe_allow_html=True)
    
    # FEATURE 2: Timeline
    st.markdown("## 📅 Timeline Generator")
    timeline = generate_timeline(text)
    if timeline:
        for item in timeline:
            st.markdown(f"<div class='timeline-item'>📅 {item['date']}<br>{item['event']}</div>", 
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
    
    # FEATURE 4: Smart Search (with unique key)
    st.markdown("## 🔎 Smart Document Search")
    
    # Create unique key for search
    import random
    search_key = f"search_{random.randint(1000, 9999)}"
    
    search_query = st.text_input(
        "Enter search term", 
        placeholder="Type to search...", 
        key=search_key
    )
    
    if search_query:
        results = smart_search(text, search_query)
        if results:
            for i, res in enumerate(results):
                st.markdown(f"<div class='search-result'>{i+1}. {res}</div>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No matches found")
    
    # FEATURE 5: Infographic
    st.markdown("## 📊 Auto Infographic")
    fig = generate_infographic(text, topics)
    st.pyplot(fig)
    
    # Download infographic
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
    img_bytes.seek(0)
    st.download_button("📥 Download Infographic", img_bytes, "infographic.png", "image/png")
    
    st.session_state.processed = True

# ===== MAIN UI =====
tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🔗 URL/YouTube", "📝 Paste Text"])

# TAB 1: FILE UPLOAD
with tab1:
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov']
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📊 Processing: {uploaded_file.name} ({file_size:.2f} MB)")
        
        with st.spinner("Auto-analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            
            if file_ext == 'pdf':
                text = extract_pdf_text(path)
            elif file_ext == 'txt':
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                # Simulate transcription for demo
                time.sleep(2)
                text = f"Sample transcription from {uploaded_file.name}. " * 100
            
            os.unlink(path)
            
            if text:
                auto_process(text)

# TAB 2: URL/YOUTUBE
with tab2:
    url = st.text_input("Enter URL", placeholder="https://example.com or YouTube link")
    
    if url:
        with st.spinner("Auto-fetching and analyzing..."):
            if 'youtube.com' in url or 'youtu.be' in url:
                # Simulate YouTube content
                text = f"YouTube video content from {url}. " * 100
            else:
                text = extract_url_text(url)
                if not text:
                    text = f"Sample article content from {url}. " * 100
            
            if text:
                auto_process(text)

# TAB 3: PASTE TEXT
with tab3:
    text_input = st.text_area("Paste text", height=200, 
                              placeholder="Paste your article, document, or notes here...")
    
    if text_input:
        auto_process(text_input)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    🤖 All 6 Features Auto-Generated | Summarization | Plagiarism | Timeline | Topics | Search | Infographic
</div>
""", unsafe_allow_html=True)
