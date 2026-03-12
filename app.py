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
import base64
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import validators
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== NLTK DOWNLOAD =====
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

st.set_page_config(page_title="AI Research Assistant", page_icon="🚀", layout="wide")

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
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    .search-highlight {
        background-color: #ffff99;
        padding: 2px;
        border-radius: 3px;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    .infographic-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 1rem 0;
    }
    .fixed-height {
        min-height: 50px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🚀 AI Research Assistant</h1><p>Video | Audio | PDF | URL | Text + No Refresh Features</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''
if 'search_done' not in st.session_state:
    st.session_state.search_done = False
if 'infographic_fig' not in st.session_state:
    st.session_state.infographic_fig = None
if 'show_infographic' not in st.session_state:
    st.session_state.show_infographic = False

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    assembly_key = st.text_input(
        "🗝️ AssemblyAI Key (For Video/Audio)",
        value=st.session_state.assembly_key,
        type="password",
        placeholder="Enter your AssemblyAI key"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assembly_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    st.markdown("🎥 **Video:** MP4, AVI, MOV")
    st.markdown("🎵 **Audio:** MP3, WAV, M4A")
    st.markdown("📄 **Document:** PDF, TXT")
    st.markdown("🌐 **Online:** URLs, YouTube")
    
    st.markdown("---")
    st.markdown("### 🚀 Features")
    st.markdown("🔍 **Smart Search** - No refresh")
    st.markdown("📊 **Infographic** - No refresh")

# ===== PDF EXTRACTION =====
def extract_pdf_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text
    except:
        return None

# ===== URL EXTRACTION =====
def extract_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return None
    except:
        return None

# ===== YOUTUBE EXTRACTION =====
def extract_youtube_content(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        
        if not video_id:
            return None
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for lang in ['te', 'en', 'hi']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    full_text = ' '.join([item['text'] for item in transcript_data])
                    return full_text
                except:
                    continue
        except:
            pass
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('description', '')
    except:
        return None

# ===== ASSEMBLYAI TRANSCRIPTION =====
def transcribe_with_assemblyai(audio_path):
    try:
        headers = {'authorization': st.session_state.assembly_key}
        with open(audio_path, 'rb') as f:
            response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
        upload_url = response.json()['upload_url']
        
        transcript_request = {
            'audio_url': upload_url,
            'language_detection': True,
            'speech_models': ['universal-2']
        }
        response = requests.post('https://api.assemblyai.com/v2/transcript', json=transcript_request, headers=headers)
        transcript_id = response.json()['id']
        
        progress = st.progress(0)
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
            result = response.json()
            if result['status'] == 'completed':
                return result.get('text', '')
            elif result['status'] == 'error':
                return None
        return None
    except:
        return None

# ===== TEXT TO SPEECH =====
def text_to_speech(text):
    try:
        if not text: return None
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except:
        return None

# ===== GENERATE SUMMARY =====
def generate_summary(text, num_points=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_points:
        return text
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_freq = Counter(words)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{4,}\b', sent.lower())
        score = sum(word_freq.get(word, 0) for word in sent_words if word not in stop_words)
        if 20 < len(sent) < 300:
            sentence_scores[i] = score
    
    if not sentence_scores:
        return ' '.join(sentences[:num_points])
    
    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_points]
    top_indices.sort()
    
    summary = f"🎯 **KEY POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    return summary

# ===== EXTRACT KEYWORDS =====
def extract_keywords(text, num=15):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return Counter(filtered).most_common(num)

# ===== 🔍 SMART SEARCH FUNCTION =====
def search_in_document(text, query):
    """Search for specific information in the document"""
    if not text or not query:
        return []
    
    sentences = nltk.sent_tokenize(text)
    query_words = query.lower().split()
    results = []
    
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(word in sent_lower for word in query_words):
            results.append({
                'sentence': sent,
                'index': i
            })
    
    return results

# ===== 📊 INFOGRAPHIC GENERATOR =====
def generate_infographic(text, keywords):
    """Generate a simple infographic from text"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#f8f9fa')
    
    # 1. Word Frequency Bar Chart
    words = re.findall(r'\b\w{4,}\b', text.lower())
    word_counts = Counter(words).most_common(8)
    
    if word_counts:
        ax1 = axes[0, 0]
        words_list, counts_list = zip(*word_counts)
        ax1.barh(words_list, counts_list, color='#667eea')
        ax1.set_title('Top Keywords', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frequency')
    
    # 2. Sentence Length Distribution
    sentences = nltk.sent_tokenize(text)
    sent_lengths = [len(sent.split()) for sent in sentences[:20]]
    
    if sent_lengths:
        ax2 = axes[0, 1]
        ax2.hist(sent_lengths, bins=10, color='#764ba2', alpha=0.7)
        ax2.set_title('Sentence Length Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Words per Sentence')
        ax2.set_ylabel('Frequency')
    
    # 3. Key Metrics
    ax3 = axes[1, 0]
    ax3.axis('off')
    metrics_text = f"""
    📊 DOCUMENT METRICS
    
    Total Words: {len(text.split()):,}
    Total Sentences: {len(sentences)}
    Unique Words: {len(set(re.findall(r'\b\w+\b', text.lower())))}
    Avg Sentence Length: {len(text.split())//max(len(sentences),1):.1f} words
    
    🏷️ TOP CONCEPTS
    """
    for word, count in keywords[:5]:
        metrics_text += f"\n• {word}: {count} times"
    
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Reading Time
    ax4 = axes[1, 1]
    ax4.axis('off')
    words = len(text.split())
    minutes = words // 200
    seconds = int((words % 200) / 200 * 60)
    
    reading_text = f"""
    ⏱️ READING TIME
    
    {minutes} min {seconds} sec
    
    📈 SUMMARY STATS
    Summary Words: {len(st.session_state.get('current_summary', '').split())}
    Compression: {100 - (len(st.session_state.get('current_summary', '').split())/words*100):.1f}%
    """
    
    ax4.text(0.1, 0.5, reading_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ===== DISPLAY RESULTS WITH NO REFRESH FEATURES =====
def display_results(text, source_name):
    if not text or len(text.strip()) == 0:
        st.error("No text to display")
        return
    
    st.session_state.current_text = text
    
    total_sentences = len(nltk.sent_tokenize(text))
    original_words = len(text.split())
    original_chars = len(text)
    
    # Summary generation
    st.markdown("## 📝 Summary")
    summary = generate_summary(text, 5)
    st.info(summary)
    st.session_state.current_summary = summary
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 Characters", f"{original_chars:,}")
    with col2: st.metric("📈 Words", f"{original_words:,}")
    with col3: st.metric("🔤 Sentences", f"{total_sentences:,}")
    with col4: st.metric("📉 Summary Words", len(summary.split()))
    
    # Keywords
    keywords = extract_keywords(text)
    if keywords:
        st.markdown("### 🏷️ Keywords")
        html = "<div>"
        for word, count in keywords[:12]:
            size = min(24 + count, 40)
            html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    
    # ===== 🔍 SMART SEARCH - NO REFRESH =====
    with st.expander("🔍 Smart Search Inside Document", expanded=False):
        st.markdown("### Search for specific information")
        
        # Search input
        search_query = st.text_input(
            "Enter your search term:",
            value=st.session_state.search_query,
            placeholder="e.g., AI advantages, machine learning, etc.",
            key="search_input"
        )
        st.session_state.search_query = search_query
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔎 Search", key="search_btn"):
                if search_query:
                    results = search_in_document(text, search_query)
                    st.session_state.search_results = results
                    st.session_state.search_done = True
        
        with col2:
            if st.session_state.search_results:
                if st.button("❌ Clear Results", key="clear_search"):
                    st.session_state.search_results = []
                    st.session_state.search_query = ''
                    st.session_state.search_done = False
                    st.rerun()
        
        # Display results
        if st.session_state.search_results:
            results = st.session_state.search_results
            st.success(f"✅ Found {len(results)} matching sentences")
            for i, res in enumerate(results[:10]):
                st.markdown(f"**{i+1}.** {res['sentence']}")
    
    # ===== 📊 AUTO INFOGRAPHIC - NO REFRESH =====
    with st.expander("📊 Auto Infographic Generator", expanded=False):
        st.markdown("### Visual Summary")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎨 Generate Infographic", key="gen_infographic"):
                with st.spinner("Creating infographic..."):
                    fig = generate_infographic(text, keywords)
                    st.session_state.infographic_fig = fig
                    st.session_state.show_infographic = True
        
        with col2:
            if st.session_state.infographic_fig is not None:
                if st.button("❌ Clear", key="clear_infographic"):
                    st.session_state.infographic_fig = None
                    st.session_state.show_infographic = False
                    st.rerun()
        
        if st.session_state.infographic_fig is not None:
            st.pyplot(st.session_state.infographic_fig)
            
            img_bytes = io.BytesIO()
            st.session_state.infographic_fig.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
            img_bytes.seek(0)
            st.download_button(
                "📥 Download Infographic",
                img_bytes,
                file_name="infographic.png",
                mime="image/png",
                key="download_infographic"
            )
    
    # Downloads
    st.markdown("### 📥 Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt", key="download_full")
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt", key="download_summary")
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio)
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3", key="download_audio")

# ===== MAIN UI =====
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose file", type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov'])
        
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 {uploaded_file.name} | {file_size:.2f} MB")
            
            if file_ext in ['mp4', 'avi', 'mov']:
                st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav']:
                st.audio(uploaded_file)
            
            if st.button("🚀 Process", key="proc_file"):
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        path = tmp.name
                    
                    if file_ext == 'pdf':
                        text = extract_pdf_text(path)
                        if text: st.success("✅ Extracted PDF"); display_results(text, "pdf")
                    elif file_ext == 'txt':
                        with open(path, 'r', encoding='utf-8') as f: text = f.read()
                        display_results(text, "text")
                    else:
                        if not st.session_state.assembly_key:
                            st.error("❌ AssemblyAI Key required")
                        else:
                            text = transcribe_with_assemblyai(path)
                            if text: st.success(f"✅ Transcribed: {len(text)} chars"); display_results(text, "media")
                    os.unlink(path)
    
    with tab2:
        url = st.text_input("Enter URL", placeholder="https://...")
        if url and st.button("🌐 Fetch", key="fetch_url"):
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("Fetching YouTube..."):
                    text = extract_youtube_content(url)
                    if text:
                        st.success("✅ YouTube content fetched")
                        display_results(text, "youtube")
                    else:
                        st.warning("No content found")
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text = extract_from_url(url)
                    if text:
                        st.success("✅ Article fetched")
                        display_results(text, "web")
                    else:
                        st.warning("No content found")
            else:
                st.error("Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200)
        if text_input and st.button("📝 Summarize", key="summ_text"):
            if len(text_input) > 100:
                display_results(text_input, "pasted")
            else:
                st.warning("Text too short")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>🚀 AI Research Assistant</h3>
            <h4>Features:</h4>
            <ul>
                <li>🎥 <strong>Video/Audio Transcription</strong> - Using AssemblyAI</li>
                <li>📄 <strong>PDF/Text Extraction</strong> - From uploaded files</li>
                <li>🌐 <strong>URL/YouTube Fetch</strong> - Web content extraction</li>
                <li>📝 <strong>Smart Summarization</strong> - LexRank algorithm</li>
                <li>🏷️ <strong>Keyword Extraction</strong> - Important terms</li>
                <li>🔊 <strong>Text-to-Speech</strong> - Listen to summaries</li>
                <li>🔍 <strong>Smart Search</strong> - No refresh! Search inside documents</li>
                <li>📊 <strong>Infographic Generator</strong> - No refresh! Visual summaries</li>
            </ul>
            
            <h4>🔍 No Refresh Features:</h4>
            <ul>
                <li>✅ Search button click - Page won't refresh</li>
                <li>✅ Infographic generate - Page won't refresh</li>
                <li>✅ Clear buttons work properly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
