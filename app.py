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

# Download NLTK data
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

st.set_page_config(page_title="Video/Text Summarizer", page_icon="🎥", layout="wide")

# Custom CSS
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
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎥 Video/Text Summarizer</h1><p>Upload video, audio, PDF, or paste URL/text</p></div>", unsafe_allow_html=True)

# Session state
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''

# Sidebar
with st.sidebar:
    st.header("🔑 API Key")
    assembly_key = st.text_input("AssemblyAI Key", type="password")
    if st.button("Save Key"):
        st.session_state.assembly_key = assembly_key
        st.success("Key saved!")

# PDF extraction
def extract_pdf_text(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except:
        return None

# URL extraction
def extract_url_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return ' '.join(p.get_text() for p in soup.find_all('p'))
    except:
        return None

# YouTube extraction
def get_youtube_text(url):
    try:
        video_id = url.split('v=')[-1].split('&')[0] if 'youtube.com' in url else url.split('youtu.be/')[-1]
        
        # Try captions first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return ' '.join([t['text'] for t in transcript]), "Captions"
        except:
            # Get description
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('description', ''), "Description"
    except:
        return None, None

# AssemblyAI transcription
def transcribe_audio(audio_path):
    if not st.session_state.assembly_key:
        return "Please add AssemblyAI key in sidebar"
    
    headers = {'authorization': st.session_state.assembly_key}
    
    # Upload
    with open(audio_path, 'rb') as f:
        res = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
    upload_url = res.json()['upload_url']
    
    # Transcribe
    res = requests.post('https://api.assemblyai.com/v2/transcript', 
                       headers=headers, 
                       json={'audio_url': upload_url, 'speech_models': ['universal-2']})
    transcript_id = res.json()['id']
    
    # Poll
    progress = st.progress(0)
    for i in range(30):
        time.sleep(2)
        progress.progress((i+1)*3)
        res = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        result = res.json()
        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'error':
            return None
    return None

# Generate summary
def get_summary(text, num_sentences=5):
    if not text or len(text) < 100:
        return text
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(s) for s in summary)

# Get keywords
def get_keywords(text, count=10):
    words = re.findall(r'\b\w{4,}\b', text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(count)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🔗 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])

with tab1:
    file = st.file_uploader("Upload file", type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov'])
    
    if file:
        ext = file.name.split('.')[-1].lower()
        st.video(file) if ext in ['mp4', 'avi', 'mov'] else st.audio(file)
        
        if st.button("Process File"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getvalue())
                    path = tmp.name
                
                if ext == 'pdf':
                    text = extract_pdf_text(path)
                elif ext == 'txt':
                    with open(path, 'r') as f:
                        text = f.read()
                else:
                    text = transcribe_audio(path)
                
                os.unlink(path)
                
                if text:
                    st.session_state['current_text'] = text
                    summary = get_summary(text)
                    keywords = get_keywords(text)
                    
                    st.markdown("## 📝 Summary")
                    st.info(summary)
                    
                    st.markdown("## 🌍 Current Affairs Format")
                    st.markdown(f"**Today's Headlines** - {datetime.now().strftime('%B %d, %Y')}")
                    for i, sent in enumerate(text.split('.')[:5]):
                        st.markdown(f"• {sent.strip()}.")
                    
                    st.markdown("## 🏷️ Keywords")
                    html = ""
                    for word, count in keywords:
                        html += f"<span class='keyword-tag'>{word}</span> "
                    st.markdown(html, unsafe_allow_html=True)
                    
                    audio = text_to_speech(summary)
                    if audio:
                        st.audio(audio)
                        st.download_button("Download Audio", audio, "summary.mp3")

with tab2:
    url = st.text_input("Enter URL")
    
    if url and st.button("Fetch"):
        with st.spinner("Fetching..."):
            if 'youtube.com' in url or 'youtu.be' in url:
                text, source = get_youtube_text(url)
                if text:
                    st.success(f"✅ Got content from {source}")
                else:
                    st.error("Could not get video content")
            else:
                text = extract_url_text(url)
            
            if text:
                st.session_state['current_text'] = text
                summary = get_summary(text)
                keywords = get_keywords(text)
                
                st.markdown("## 📝 Summary")
                st.info(summary)
                
                st.markdown("## 🌍 Current Affairs Format")
                for i, sent in enumerate(text.split('.')[:5]):
                    st.markdown(f"• {sent.strip()}.")
                
                st.markdown("## 🏷️ Keywords")
                html = ""
                for word, count in keywords:
                    html += f"<span class='keyword-tag'>{word}</span> "
                st.markdown(html, unsafe_allow_html=True)
                
                audio = text_to_speech(summary)
                if audio:
                    st.audio(audio)
                    st.download_button("Download Audio", audio, "summary.mp3")

with tab3:
    text_input = st.text_area("Paste your text here", height=200)
    
    if text_input and st.button("Summarize"):
        if len(text_input) > 200:
            st.session_state['current_text'] = text_input
            summary = get_summary(text_input)
            keywords = get_keywords(text_input)
            
            st.markdown("## 📝 Summary")
            st.info(summary)
            
            st.markdown("## 🌍 Current Affairs Format")
            for i, sent in enumerate(text_input.split('.')[:5]):
                st.markdown(f"• {sent.strip()}.")
            
            st.markdown("## 🏷️ Keywords")
            html = ""
            for word, count in keywords:
                html += f"<span class='keyword-tag'>{word}</span> "
            st.markdown(html, unsafe_allow_html=True)
            
            audio = text_to_speech(summary)
            if audio:
                st.audio(audio)
                st.download_button("Download Audio", audio, "summary.mp3")

with tab4:
    st.markdown("""
    ## 📌 How to Use
    1. **Add AssemblyAI Key** in sidebar (free from assemblyai.com)
    2. **Upload files** or **paste URL/text**
    3. Get summary, keywords, and audio
    4. Download results
    
    ## ✨ Features
    - 🎥 Video to text
    - 🎵 Audio transcription
    - 📄 PDF extraction
    - 🌐 URL fetching
    - 📝 Text summarization
    - 🏷️ Keyword extraction
    - 🔊 Audio download
    """)

if __name__ == "__main__":
    main()
