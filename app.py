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

# ===== CUSTOM CSS - ORANGE THEME =====
st.markdown("""
<style>
    /* Main header - Orange gradient */
    .main-header {
        background: linear-gradient(135deg, #ff8c42, #ff5e3a);
        padding: 2rem;
        border-radius: 30px 30px 30px 30px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(255, 94, 58, 0.3);
    }
    
    /* Section cards - Rounded, no rectangles */
    .section-card {
        background: #fff5e6;
        padding: 1.5rem;
        border-radius: 25px;
        border-left: none;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(255, 94, 58, 0.1);
    }
    
    /* Keyword tags - Orange */
    .keyword-tag {
        background: linear-gradient(135deg, #ff8c42, #ff5e3a);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin: 0.3rem;
        font-size: 0.9rem;
        box-shadow: 0 3px 8px rgba(255, 94, 58, 0.2);
    }
    
    /* Feature tags in sidebar */
    .feature-tag {
        background: #fff5e6;
        color: #ff5e3a;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
        border: 1px solid #ff8c42;
    }
    
    /* Buttons - Orange rounded */
    .stButton > button {
        background: linear-gradient(135deg, #ff8c42, #ff5e3a);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 94, 58, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 94, 58, 0.4);
    }
    
    /* Tabs - Orange rounded */
    .stTabs [data-baseweb="tab-list"] {
        background: #fff5e6;
        padding: 0.5rem;
        border-radius: 50px;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ff5e3a;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff8c42, #ff5e3a);
        color: white;
    }
    
    /* Slider container */
    .slider-container {
        background: #fff5e6;
        padding: 1.5rem;
        border-radius: 30px;
        margin: 1rem 0;
        border: none;
    }
    
    /* Metric boxes - Rounded */
    .metric-box {
        background: #fff5e6;
        padding: 1.2rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 5px 10px rgba(255, 94, 58, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #fff9f0;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #ff5e3a;
    }
    
    /* Feature cards in sidebar */
    .sidebar-feature {
        background: white;
        padding: 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        border-left: none;
        box-shadow: 0 3px 10px rgba(255, 94, 58, 0.1);
    }
    
    /* Plagiarism indicators - Rounded */
    .plagiarism-low { 
        background: #d4edda; 
        color: #155724; 
        padding: 0.8rem; 
        border-radius: 20px; 
        margin: 0.3rem 0;
    }
    .plagiarism-medium { 
        background: #fff3cd; 
        color: #856404; 
        padding: 0.8rem; 
        border-radius: 20px; 
        margin: 0.3rem 0;
    }
    .plagiarism-high { 
        background: #f8d7da; 
        color: #721c24; 
        padding: 0.8rem; 
        border-radius: 20px; 
        margin: 0.3rem 0;
    }
    
    /* Timeline items - Rounded */
    .timeline-item { 
        background: #fff5e6; 
        padding: 0.8rem 1.2rem; 
        border-radius: 20px; 
        margin: 0.5rem 0; 
        border-left: none;
    }
    
    /* Timestamp items - Rounded */
    .timestamp-item { 
        background: #fff5e6; 
        padding: 0.6rem 1rem; 
        border-radius: 15px; 
        margin: 0.3rem 0; 
    }
    
    /* Moment highlight - Rounded */
    .moment-highlight { 
        background: #fff3cd; 
        padding: 0.6rem 1rem; 
        border-radius: 15px; 
        margin: 0.3rem 0; 
        border-left: none;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER - Orange =====
st.markdown("<div class='main-header'><h1>🎤 Audio to Text Summarizer</h1><p>Transform your audio, video, PDF, and text into smart summaries</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'slider_value' not in st.session_state:
    st.session_state.slider_value = 5

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assembly_key,
        type="password"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assembly_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported")
    st.markdown("🎥 Video: MP4, AVI, MOV")
    st.markdown("🎵 Audio: MP3, WAV, M4A")
    st.markdown("📄 PDF, TXT")
    st.markdown("🌐 URLs, YouTube")
    
    # ===== 5 FEATURES IN SIDEBAR =====
    st.markdown("---")
    st.markdown("### 🚀 5 Features")
    
    with st.expander("🔍 Plagiarism Checker", expanded=True):
        st.markdown("<div class='sidebar-feature'>Check text originality</div>", unsafe_allow_html=True)
    
    with st.expander("📅 Timeline Generator", expanded=True):
        st.markdown("<div class='sidebar-feature'>Convert to timeline</div>", unsafe_allow_html=True)
    
    with st.expander("🎯 Topic Detection", expanded=True):
        st.markdown("<div class='sidebar-feature'>Identify main topics</div>", unsafe_allow_html=True)
    
    with st.expander("⏱️ Timestamp Summary", expanded=True):
        st.markdown("<div class='sidebar-feature'>Time-wise summary</div>", unsafe_allow_html=True)
    
    with st.expander("🎬 Key Moments", expanded=True):
        st.markdown("<div class='sidebar-feature'>Important moments</div>", unsafe_allow_html=True)

# ===== ORIGINAL FUNCTIONS =====
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
            title = soup.title.string if soup.title else "Article"
            if text and len(text) > 200:
                return text, title
        return None, None
    except:
        return None, None

def extract_youtube_content(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        if not video_id:
            return None, None
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for lang in ['te', 'en', 'hi']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    full_text = ' '.join([item['text'] for item in transcript_data])
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(url, download=False)
                        title = info.get('title', 'YouTube Video')
                    return full_text, f"YouTube: {title}"
                except:
                    continue
        except:
            pass
        
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            if description:
                return description, f"YouTube: {title}"
        return None, None
    except:
        return None, None

def transcribe_with_assemblyai(audio_path):
    try:
        headers = {'authorization': st.session_state.assemblyai_key}
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

def generate_summary(text, num_points):
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
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    return summary

# ===== 5 FEATURES FUNCTIONS =====
def check_plagiarism(text):
    common_phrases = ["according to", "research shows", "studies indicate", "as a result", 
                      "in conclusion", "for example", "such as", "due to"]
    matches = sum(1 for phrase in common_phrases if phrase in text.lower())
    score = min(100, matches * 12)
    if score < 30: return score, "Low", "plagiarism-low"
    elif score < 60: return score, "Medium", "plagiarism-medium"
    else: return score, "High", "plagiarism-high"

def generate_timeline(text):
    sentences = nltk.sent_tokenize(text)
    timeline = []
    year_pattern = r'\b(19|20)\d{2}\b'
    for sent in sentences[:15]:
        years = re.findall(year_pattern, sent)
        if years:
            timeline.append(f"📅 {years[0]} → {sent[:80]}...")
    return timeline[:5]

def detect_topics(text):
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    word_freq = Counter(words)
    common = {'The', 'This', 'That', 'These', 'Those'}
    return [(w, c) for w, c in word_freq.most_common(8) if w not in common and len(w) > 3]

def generate_timestamps(text, duration=10):
    sentences = nltk.sent_tokenize(text)
    timestamps = []
    interval = duration / min(8, len(sentences))
    for i in range(min(8, len(sentences))):
        mins = int(i * interval)
        secs = int((i * interval - mins) * 60)
        timestamps.append(f"⏱️ {mins:02d}:{secs:02d} → {sentences[i][:70]}...")
    return timestamps

def detect_key_moments(text):
    sentences = nltk.sent_tokenize(text)
    keywords = ['important', 'key', 'significant', 'crucial', 'vital', 'breakthrough']
    moments = []
    for i, sent in enumerate(sentences[:20]):
        score = sum(1 for k in keywords if k in sent.lower())
        if score > 0:
            mins = (i * 10) // 60
            secs = (i * 10) % 60
            moments.append(f"✨ {mins:02d}:{secs:02d} - {sent[:80]}...")
    return moments[:5]

# ===== DISPLAY RESULTS =====
def display_results(text, source_name):
    if not text or len(text.strip()) == 0:
        st.error("No text to display")
        return
    
    st.session_state.current_text = text
    
    total_sentences = len(nltk.sent_tokenize(text))
    original_words = len(text.split())
    original_chars = len(text)
    
    # Slider
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        if total_sentences < 3:
            st.warning(f"⚠️ Only {total_sentences} sentence(s)")
            num_points = total_sentences
        else:
            max_val = min(30, total_sentences)
            num_points = st.slider("Number of summary sentences:", 3, max_val, st.session_state.slider_value, key="main_slider")
            st.session_state.slider_value = num_points
    with col2:
        st.metric("Total", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate summary
    if total_sentences < 3:
        summary = text
        summary_words = original_words
    else:
        summary = generate_summary(text, num_points)
        summary_words = len(summary.split())
    
    # Calculate reduction
    if original_words > 0:
        reduction = int((1 - summary_words/original_words) * 100)
        reduction = max(0, min(100, reduction))
    else:
        reduction = 0
    
    # Summary
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.markdown(f"<div class='metric-box'><b>Characters</b><br><h2>{original_chars:,}</h2></div>", unsafe_allow_html=True)
    with col2: 
        st.markdown(f"<div class='metric-box'><b>Words</b><br><h2>{original_words:,}</h2></div>", unsafe_allow_html=True)
    with col3: 
        st.markdown(f"<div class='metric-box'><b>Sentences</b><br><h2>{total_sentences:,}</h2></div>", unsafe_allow_html=True)
    with col4: 
        st.markdown(f"<div class='metric-box'><b>Reduced</b><br><h2>{reduction}%</h2></div>", unsafe_allow_html=True)
    
    # Keywords
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    if words:
        keywords = Counter(words).most_common(10)
        st.markdown("### 🏷️ Keywords")
        html = "<div>"
        for word, count in keywords[:8]:
            html += f"<span class='keyword-tag'>{word} ({count})</span> "
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    
    # Downloads
    st.markdown("### 📥 Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio)
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3")
    
    # ===== 5 FEATURES =====
    st.markdown("---")
    st.markdown("## 🚀 5 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature 1: Plagiarism
        st.markdown("### 🔍 Plagiarism Checker")
        score, level, css = check_plagiarism(text)
        st.markdown(f"<div class='{css}'><b>Score:</b> {score}% - {level} Risk</div>", unsafe_allow_html=True)
        
        # Feature 2: Timeline
        st.markdown("### 📅 Timeline Generator")
        timeline = generate_timeline(text)
        if timeline:
            for item in timeline:
                st.markdown(f"<div class='timeline-item'>{item}</div>", unsafe_allow_html=True)
        else:
            st.info("No timeline events detected")
        
        # Feature 3: Topics
        st.markdown("### 🎯 Topic Detection")
        topics = detect_topics(text)
        if topics:
            topic_html = ""
            for topic, count in topics:
                topic_html += f"<span class='keyword-tag'>{topic}</span> "
            st.markdown(topic_html, unsafe_allow_html=True)
        else:
            st.info("No topics detected")
    
    with col2:
        # Feature 4: Timestamps
        st.markdown("### ⏱️ Timestamp Summary")
        duration = st.slider("Video duration (minutes)", 5, 30, 10, key="duration_slider")
        timestamps = generate_timestamps(text, duration)
        for ts in timestamps:
            st.markdown(f"<div class='timestamp-item'>{ts}</div>", unsafe_allow_html=True)
        
        # Feature 5: Key Moments
        st.markdown("### 🎬 Key Moments")
        moments = detect_key_moments(text)
        if moments:
            for moment in moments:
                st.markdown(f"<div class='moment-highlight'>{moment}</div>", unsafe_allow_html=True)
        else:
            st.info("No key moments detected")

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
                        if not st.session_state.assemblyai_key:
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
                    text, title = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        display_results(text, "youtube")
                    else:
                        st.warning("No content found")
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text:
                        st.success(f"✅ {title}")
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
            <h3>📌 How to Use</h3>
            <ol>
                <li>Add AssemblyAI Key (for video/audio)</li>
                <li>Upload file, paste URL, or enter text</li>
                <li>Get summary, stats, keywords, downloads</li>
                <li>Plus 5 advanced features below!</li>
            </ol>
            
            <h3>🚀 5 Features</h3>
            <ul>
                <li>🔍 Plagiarism Checker</li>
                <li>📅 Timeline Generator</li>
                <li>🎯 Topic Detection</li>
                <li>⏱️ Timestamp Summary</li>
                <li>🎬 Key Moments Detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
