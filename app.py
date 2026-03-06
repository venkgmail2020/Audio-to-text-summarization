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

# ===== GEMINI AI IMPORTS =====
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ google-generativeai not installed. Run: pip install google-generativeai")

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

st.set_page_config(page_title="Audio to Text Summarizer Using NLP", page_icon="🎤", layout="wide")

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .stApp { background: white !important; }
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label { color: black !important; }
    .stTextInput > div > div > input {
        background: white !important; color: black !important; border: 1px solid #ccc !important;
        border-radius: 8px !important;
    }
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        padding: 2rem !important; border-radius: 15px !important; color: white !important;
        text-align: center !important; margin-bottom: 2rem !important;
    }
    .main-header h1, .main-header p { color: white !important; }
    .section-card {
        background: #f8f9fa !important; padding: 1.5rem !important; border-radius: 10px !important;
        border-left: 5px solid #ff6b6b !important; margin: 1rem 0 !important; color: black !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important; color: white !important;
        border: none !important; padding: 0.5rem 1.5rem !important; border-radius: 25px !important;
        font-weight: bold !important; width: 100% !important;
    }
    .chat-container {
        background: #f8f9fa !important; border-radius: 15px !important; padding: 20px !important;
        min-height: 400px !important; max-height: 500px !important; margin-bottom: 20px !important;
        border: 1px solid #ddd !important; overflow-y: auto !important;
    }
    .user-message {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important; color: white !important;
        padding: 12px 18px !important; border-radius: 20px 20px 5px 20px !important;
        margin: 10px 0 10px auto !important; max-width: 80% !important; width: fit-content !important;
        float: right !important; clear: both !important;
    }
    .bot-message {
        background: #e9ecef !important; color: black !important; padding: 12px 18px !important;
        border-radius: 20px 20px 20px 5px !important; margin: 10px auto 10px 0 !important;
        max-width: 80% !important; width: fit-content !important; float: left !important;
        clear: both !important; border: 1px solid #dee2e6 !important;
    }
    .welcome-message { text-align: center !important; padding: 80px 20px !important; color: #666 !important; }
    .welcome-message h2 { color: #ff6b6b !important; font-size: 2.5em !important; }
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; }
    [data-testid="stSidebar"] * { color: black !important; }
    .clearfix::after { content: ""; clear: both; display: table; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class='main-header'>
    <h1>🎤 Audio to Text Summarizer Using NLP</h1>
    <p style='font-size: 1.2rem;'>Transform Audio | Video | PDF | Text | URL into Smart Summaries</p>
</div>
""", unsafe_allow_html=True)

# ===== SESSION STATE INITIALIZATION =====
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ''
if 'current_text' not in st.session_state:
    st.session_state.current_text = ''
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = ''
if 'slider_value' not in st.session_state:
    st.session_state.slider_value = 5
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔐 API Configuration")
    
    assembly_key = st.text_input(
        "🗝️ AssemblyAI Key",
        value=st.session_state.assemblyai_key,
        type="password",
        placeholder="Enter your AssemblyAI key"
    )
    
    gemini_key = st.text_input(
        "🤖 Google Gemini Key (Required for AI Chat)",
        value=st.session_state.gemini_key,
        type="password",
        placeholder="Paste your new Gemini API key here"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.session_state.gemini_key = gemini_key
        st.success("✅ Keys saved! Now use AI Chat tab.")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    st.markdown("🎥 **Video:** MP4, AVI, MOV")
    st.markdown("🎵 **Audio:** MP3, WAV, M4A")
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
            title = soup.title.string if soup.title else "Article"
            if text and len(text) > 200:
                return text, title
        return None, None
    except:
        return None, None

# ===== YOUTUBE EXTRACTION =====
def extract_youtube_content(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        if not video_id:
            return None, None, None
        
        # Try transcript first
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
                    return full_text, f"YouTube: {title}", "transcript"
                except:
                    continue
        except:
            pass
        
        # Fallback to description
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', '')
            if description:
                return description, f"YouTube: {title}", "description"
        return None, None, None
    except:
        return None, None, None

# ===== ASSEMBLYAI TRANSCRIPTION =====
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
    
    summary = f"🎯 **KEY POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    return summary

# ===== GEMINI AI RESPONSE (REAL AI) =====
def get_gemini_response(user_input, context=""):
    """Real AI response using Google Gemini"""
    try:
        # Get API key from session state
        api_key = st.session_state.get('gemini_key', '')
        
        if not api_key:
            return "⚠️ Please add your Google Gemini API key in the sidebar first!"
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the CORRECT model name
        model = genai.GenerativeModel('models/gemini-2.0-flash')  # ← FIXED!
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant for a Text Summarizer app called "Audio to Text Summarizer Using NLP".
        
Current context from user's uploaded content: {context[:1000] if context else 'No content uploaded yet'}

The app can:
- Transcribe audio/video files using AssemblyAI
- Extract text from PDFs and URLs
- Generate summaries using LexRank algorithm
- Show keywords from the text
- Convert text to speech

User question: {user_input}

Provide a helpful, friendly, and accurate answer. Be conversational and use emojis occasionally.
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your Gemini API key."

# ===== DISPLAY RESULTS =====
def display_results(text, source_name):
    if not text or len(text.strip()) == 0:
        st.error("No text to display")
        return
    
    st.session_state.current_text = text
    total_sentences = len(nltk.sent_tokenize(text))
    original_words = len(text.split())
    original_chars = len(text)
    
    st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if total_sentences < 3:
            st.warning(f"⚠️ Only {total_sentences} sentence(s)")
            num_points = total_sentences
        else:
            max_val = min(30, total_sentences)
            num_points = st.slider(
                "📊 Number of summary sentences:",
                min_value=3, max_value=max_val,
                value=st.session_state.slider_value, key="main_slider"
            )
            st.session_state.slider_value = num_points
    
    with col2:
        st.metric("Total", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if total_sentences <= num_points:
        summary = text
        summary_words = original_words
        st.info(f"ℹ️ Text has only {total_sentences} sentences. Showing full text.")
    else:
        summary = generate_summary(text, num_points)
        summary_words = len(summary.split())
    
    st.session_state.current_summary = summary
    
    if original_words > 0 and summary_words < original_words:
        reduction = int((1 - summary_words/original_words) * 100)
    else:
        reduction = 0
    
    # Summary Section
    st.markdown("## 📝 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 Characters", f"{original_chars:,}")
    with col2: st.metric("📈 Words", f"{original_words:,}")
    with col3: st.metric("🔤 Sentences", f"{total_sentences:,}")
    with col4: st.metric("📉 Reduced", f"{reduction}%")
    
    # Current Affairs Section
    with st.expander("🌍 View as Current Affairs", expanded=False):
        st.markdown("### 📰 Today's Headlines")
        sentences = nltk.sent_tokenize(text)
        for i, sent in enumerate(sentences[:10]):
            if len(sent) > 30:
                st.markdown(f"• {sent}")
        
        current_text = f"CURRENT AFFAIRS - {datetime.now().strftime('%B %d, %Y')}\n\n"
        for i, sent in enumerate(sentences[:15]):
            current_text += f"{i+1}. {sent}\n\n"
        
        st.download_button(
            "📥 Download as Current Affairs",
            current_text,
            file_name=f"current_affairs_{datetime.now().strftime('%Y%m%d')}.txt"
        )
    
    # Reading Time
    minutes = original_words // 200
    seconds = int((original_words % 200) / 200 * 60)
    st.metric("⏳ Reading Time", f"{minutes} min {seconds} sec")
    
    # Downloads
    st.markdown("### 📥 Downloads")
    col1, col2, col3 = st.columns(3)
    with col1: st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    with col2: st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3", "audio/mp3")
    
    # Keywords
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    if words:
        keywords = Counter(words).most_common(15)
        st.markdown("### 🏷️ Keywords")
        html = "<div>"
        for word, count in keywords[:12]:
            size = min(24 + count, 40)
            html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# ===== MAIN UI =====
def main():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "🤖 AI Chat", "ℹ️ Help"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose file", type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'pdf', 'txt'])
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 {uploaded_file.name} | {file_size:.2f} MB")
            
            if file_ext in ['mp4', 'avi', 'mov']: st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav', 'm4a']: st.audio(uploaded_file)
            
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
                    text, title, content_type = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        if content_type == "description": st.info("ℹ️ Showing video description (no captions available)")
                        display_results(text, "youtube")
                    else: st.warning("No content found")
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text: st.success(f"✅ {title}"); display_results(text, "web")
                    else: st.warning("No content found")
            else: st.error("Invalid URL")
    
    with tab3:
        text_input = st.text_area("Paste text", height=200)
        if text_input and st.button("📝 Summarize", key="summ_text"):
            if len(text_input) > 100: display_results(text_input, "pasted")
            else: st.warning("Text too short")
    
    with tab4:
        display_chatbot()
    
    with tab5:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>Get API Keys:</strong>
                    <ul>
                        <li><a href='https://www.assemblyai.com/' target='_blank'>AssemblyAI</a> - for transcription</li>
                        <li><a href='https://aistudio.google.com/' target='_blank'>Google Gemini</a> - for AI Chat</li>
                    </ul>
                </li>
                <li><strong>Choose Input:</strong> Upload file, paste URL, or enter text</li>
                <li><strong>Adjust Summary:</strong> Use slider to control summary length</li>
                <li><strong>Ask AI:</strong> Use AI Chat tab for questions</li>
                <li><strong>Download:</strong> Get text, summary, or audio</li>
            </ol>
            <h3>✨ Features</h3>
            <ul>
                <li>🎤 <strong>Audio to Text</strong> - Transcribe audio/video files</li>
                <li>📊 <strong>Smart Summaries</strong> - Extract key points</li>
                <li>🌍 <strong>Current Affairs</strong> - View as news headlines</li>
                <li>🤖 <strong>AI Chatbot (Gemini)</strong> - Ask anything!</li>
                <li>📥 <strong>Download</strong> - Text, summary, audio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
