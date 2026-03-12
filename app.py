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

# ===== NLTK DOWNLOAD =====
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
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🎥 Video/Text Summarizer</h1><p>Upload video, audio, PDF, or paste URL/text</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assembly_key' not in st.session_state:
    st.session_state.assembly_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''

# ===== SIDEBAR =====
with st.sidebar:
    st.header("🔑 API Configuration")
    assembly_key = st.text_input("AssemblyAI Key (for video/audio)", type="password", value=st.session_state.assembly_key)
    if st.button("Save Key", use_container_width=True):
        st.session_state.assembly_key = assembly_key
        st.success("✅ Key saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    st.markdown("🎥 **Video:** MP4, AVI, MOV")
    st.markdown("🎵 **Audio:** MP3, WAV")
    st.markdown("📄 **Document:** PDF, TXT")
    st.markdown("🌐 **Online:** URLs, YouTube")

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
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

# ===== URL EXTRACTION =====
def extract_url_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            return text.strip()
        return None
    except Exception as e:
        st.error(f"URL extraction error: {e}")
        return None

# ===== YOUTUBE EXTRACTION =====
def get_youtube_text(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        
        if not video_id:
            return None, "Invalid YouTube URL"
        
        # Try captions first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['te', 'en', 'hi'])
            text = ' '.join([t['text'] for t in transcript])
            return text, "Captions"
        except:
            pass
        
        # Get description
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', '')
                description = info.get('description', '')
                if description:
                    return description, f"Description - {title}"
        except:
            pass
        
        return None, "No content available"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===== ASSEMBLYAI TRANSCRIPTION =====
def transcribe_audio(audio_path):
    api_key = st.session_state.get('assembly_key', '').strip()
    
    if not api_key:
        st.error("❌ Please add AssemblyAI key in sidebar")
        return None
    
    headers = {'authorization': api_key}
    
    try:
        # Upload file
        with st.spinner("📤 Uploading to AssemblyAI..."):
            with open(audio_path, 'rb') as f:
                upload_response = requests.post(
                    'https://api.assemblyai.com/v2/upload',
                    headers=headers,
                    data=f,
                    timeout=60
                )
            
            if upload_response.status_code != 200:
                st.error(f"Upload failed: {upload_response.status_code}")
                return None
            
            upload_url = upload_response.json().get('upload_url')
            if not upload_url:
                st.error("No upload URL received")
                return None
        
        # Request transcription
        with st.spinner("⏳ Requesting transcription..."):
            transcript_request = {
                'audio_url': upload_url,
                'language_detection': True,
                'speech_models': ['universal-2']
            }
            
            transcribe_response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                json=transcript_request,
                headers=headers
            )
            
            if transcribe_response.status_code != 200:
                st.error(f"Transcription request failed: {transcribe_response.status_code}")
                return None
            
            transcript_id = transcribe_response.json().get('id')
            if not transcript_id:
                st.error("No transcript ID received")
                return None
        
        # Poll for results
        progress = st.progress(0)
        status = st.empty()
        
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            status.text(f"⏳ Transcribing... {i*2}s")
            
            poll_response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
            
            if poll_response.status_code == 200:
                result = poll_response.json()
                status_text = result.get('status')
                
                if status_text == 'completed':
                    progress.progress(100)
                    status.text("✅ Complete!")
                    return result.get('text', '')
                elif status_text == 'error':
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"Transcription error: {error_msg}")
                    return None
        
        st.error("Transcription timeout")
        return None
        
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout - please try again")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error - check your internet")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ===== TEXT TO SPEECH =====
def text_to_speech(text):
    try:
        if not text:
            return None
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")
        return None

# ===== GENERATE SUMMARY =====
def get_summary(text, num_sentences=5):
    if not text or len(text) < 100:
        return text
    
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join(str(s) for s in summary)
    except Exception as e:
        st.warning(f"Summary generation failed: {e}")
        return text

# ===== GET KEYWORDS =====
def get_keywords(text, count=10):
    try:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words = [w for w in words if w not in stopwords]
        return Counter(words).most_common(count)
    except:
        return []

# ===== MAIN TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🔗 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])

with tab1:
    file = st.file_uploader("Upload file", type=['mp4', 'mp3', 'wav', 'pdf', 'txt', 'avi', 'mov'])
    
    if file:
        ext = file.name.split('.')[-1].lower()
        col1, col2 = st.columns(2)
        with col1:
            if ext in ['mp4', 'avi', 'mov']:
                st.video(file)
            elif ext in ['mp3', 'wav']:
                st.audio(file)
        with col2:
            file_size = len(file.getvalue()) / (1024 * 1024)
            st.info(f"📊 File: {file.name}\nSize: {file_size:.2f} MB")
        
        if st.button("🚀 Process File", use_container_width=True):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getvalue())
                    path = tmp.name
                
                if ext == 'pdf':
                    text = extract_pdf_text(path)
                elif ext == 'txt':
                    with open(path, 'r', encoding='utf-8') as f:
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
                    st.markdown(f"**Today's Headlines - {datetime.now().strftime('%B %d, %Y')}**")
                    sentences = text.split('.')
                    for i, sent in enumerate(sentences[:5]):
                        if sent.strip():
                            st.markdown(f"• {sent.strip()}.")
                    
                    if keywords:
                        st.markdown("## 🏷️ Keywords")
                        html = ""
                        for word, count in keywords:
                            html += f"<span class='keyword-tag'>{word}</span> "
                        st.markdown(html, unsafe_allow_html=True)
                    
                    audio = text_to_speech(summary)
                    if audio:
                        st.audio(audio)
                        st.download_button("🔊 Download Audio", audio, "summary.mp3")

with tab2:
    url = st.text_input("Enter URL", placeholder="https://example.com or YouTube link")
    
    if url and st.button("🌐 Fetch", use_container_width=True):
        with st.spinner("Fetching content..."):
            if 'youtube.com' in url or 'youtu.be' in url:
                text, source = get_youtube_text(url)
                if text:
                    st.success(f"✅ Got content from {source}")
                else:
                    st.error("Could not get video content")
            elif validators.url(url):
                text = extract_url_text(url)
                if text:
                    st.success("✅ Content fetched successfully")
                else:
                    st.error("Could not fetch content")
            else:
                st.error("Invalid URL")
                text = None
            
            if text:
                st.session_state['current_text'] = text
                summary = get_summary(text)
                keywords = get_keywords(text)
                
                st.markdown("## 📝 Summary")
                st.info(summary)
                
                st.markdown("## 🌍 Current Affairs Format")
                sentences = text.split('.')
                for i, sent in enumerate(sentences[:5]):
                    if sent.strip():
                        st.markdown(f"• {sent.strip()}.")
                
                if keywords:
                    st.markdown("## 🏷️ Keywords")
                    html = ""
                    for word, count in keywords:
                        html += f"<span class='keyword-tag'>{word}</span> "
                    st.markdown(html, unsafe_allow_html=True)
                
                audio = text_to_speech(summary)
                if audio:
                    st.audio(audio)
                    st.download_button("🔊 Download Audio", audio, "summary.mp3")

with tab3:
    text_input = st.text_area("Paste your text here", height=200, placeholder="Paste news article, document, or any text...")
    
    if text_input and st.button("📝 Summarize", use_container_width=True):
        if len(text_input) > 100:
            st.session_state['current_text'] = text_input
            summary = get_summary(text_input)
            keywords = get_keywords(text_input)
            
            st.markdown("## 📝 Summary")
            st.info(summary)
            
            st.markdown("## 🌍 Current Affairs Format")
            sentences = text_input.split('.')
            for i, sent in enumerate(sentences[:5]):
                if sent.strip():
                    st.markdown(f"• {sent.strip()}.")
            
            if keywords:
                st.markdown("## 🏷️ Keywords")
                html = ""
                for word, count in keywords:
                    html += f"<span class='keyword-tag'>{word}</span> "
                st.markdown(html, unsafe_allow_html=True)
            
            audio = text_to_speech(summary)
            if audio:
                st.audio(audio)
                st.download_button("🔊 Download Audio", audio, "summary.mp3")
        else:
            st.warning("Text too short (minimum 100 characters)")

with tab4:
    st.markdown("""
    <div class='section-card'>
        <h3>📌 How to Use</h3>
        <ol>
            <li><strong>Get AssemblyAI Key:</strong> Free from <a href='https://www.assemblyai.com/' target='_blank'>assemblyai.com</a> (for video/audio)</li>
            <li><strong>Choose Input:</strong> Upload file, paste URL, or enter text</li>
            <li><strong>Get Results:</strong> Summary, keywords, and audio</li>
            <li><strong>Download:</strong> Save summary as audio</li>
        </ol>
        
        <h3>✨ Features</h3>
        <ul>
            <li>🎥 <strong>Video to Text</strong> - Transcribe video files</li>
            <li>🎵 <strong>Audio to Text</strong> - Transcribe audio files</li>
            <li>📄 <strong>PDF Extraction</strong> - Extract text from PDFs</li>
            <li>🌐 <strong>URL/YouTube</strong> - Fetch content from web</li>
            <li>📝 <strong>Text Summarization</strong> - Get key points</li>
            <li>🏷️ <strong>Keywords</strong> - Important terms</li>
            <li>🔊 <strong>Audio Download</strong> - Listen to summaries</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
