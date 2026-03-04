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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import qrcode
from PIL import Image
from deep_translator import GoogleTranslator
import google.generativeai as genai

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

# ===== CONFIGURE GEMINI AI =====
try:
    # Try to get from secrets first, then from session state
    if 'GEMINI_API_KEY' in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    elif 'gemini_key' in st.session_state and st.session_state.gemini_key:
        genai.configure(api_key=st.session_state.gemini_key)
    else:
        st.warning("⚠️ Gemini API key not configured. Chatbot will use basic responses.")
except:
    pass

st.set_page_config(page_title="Audio to Text Summarizer Using NLP", page_icon="🎤", layout="wide")

# ===== CUSTOM CSS WITH DARK BACKGROUND =====
st.markdown("""
<style>
    /* Main background - DARK GRADIENT */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    }
    
    /* All text white */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, .stTextInput label, .stSelectbox label {
        color: white !important;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        color: white !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }
    
    /* Section cards */
    .section-card {
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 5px solid #ff6b6b !important;
        margin: 1rem 0 !important;
        color: white !important;
    }
    
    /* Keyword tags */
    .keyword-tag {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        color: white !important;
        padding: 0.3rem 0.8rem !important;
        border-radius: 20px !important;
        display: inline-block !important;
        margin: 0.2rem !important;
    }
    
    /* Metric boxes */
    .metric-box {
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(5px) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(255,107,107,0.4) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Slider */
    .slider-container {
        background: rgba(255,255,255,0.1) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.1) !important;
        padding: 0.5rem !important;
        border-radius: 10px !important;
        gap: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        color: white !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 20px 20px 5px 20px !important;
        margin: 0.5rem 0 !important;
        text-align: right !important;
        max-width: 80% !important;
        float: right !important;
        clear: both !important;
    }
    .bot-message {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 20px 20px 20px 5px !important;
        margin: 0.5rem 0 !important;
        text-align: left !important;
        max-width: 80% !important;
        float: left !important;
        clear: both !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    .chat-container {
        background: rgba(0,0,0,0.2) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        min-height: 400px !important;
        max-height: 500px !important;
        overflow-y: auto !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Success messages */
    .success-msg {
        background: rgba(40, 167, 69, 0.2) !important;
        color: #d4edda !important;
        padding: 0.5rem !important;
        border-radius: 5px !important;
        border: 1px solid #28a745 !important;
    }
    
    /* Info messages */
    .info-msg {
        background: rgba(23, 162, 184, 0.2) !important;
        color: #d1ecf1 !important;
        padding: 0.5rem !important;
        border-radius: 5px !important;
        border: 1px solid #17a2b8 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff6b6b, #556270);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class='main-header'>
    <h1>🎤 Audio to Text Summarizer Using NLP</h1>
    <p style='font-size: 1.2rem; opacity: 0.9;'>Transform Audio | Video | PDF | Text | URL into Smart Summaries</p>
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
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = None
if 'qr_code' not in st.session_state:
    st.session_state.qr_code = None
if 'wordcloud_fig' not in st.session_state:
    st.session_state.wordcloud_fig = None
if 'show_wordcloud' not in st.session_state:
    st.session_state.show_wordcloud = False
if 'show_qr' not in st.session_state:
    st.session_state.show_qr = False
if 'show_translation' not in st.session_state:
    st.session_state.show_translation = False
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔐 API Configuration")
    
    assembly_key = st.text_input(
        "🗝️ AssemblyAI Key",
        value=st.session_state.assemblyai_key,
        type="password"
    )
    
    gemini_key = st.text_input(
        "🤖 Google Gemini Key (for AI Chat)",
        value=st.session_state.gemini_key,
        type="password"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.session_state.gemini_key = gemini_key
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                st.success("✅ Gemini configured!")
            except:
                st.error("❌ Invalid Gemini key")
        else:
            st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📌 Supported Formats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🎥 **Video**\n- MP4\n- AVI\n- MOV")
        st.markdown("🎵 **Audio**\n- MP3\n- WAV\n- M4A")
    with col2:
        st.markdown("📄 **Document**\n- PDF\n- TXT")
        st.markdown("🌐 **Online**\n- URLs\n- YouTube")
    
    st.markdown("---")
    
    # History viewer
    if st.session_state.history:
        with st.expander("📜 Recent History"):
            for i, item in enumerate(st.session_state.history[-5:]):
                st.write(f"**{item['time']}**")
                st.write(f"📄 {item['source']} - {item['words']} words")
                if st.button(f"View #{i+1}", key=f"hist_{i}"):
                    st.session_state.current_summary = item['full_summary']
                st.divider()

# ===== GEMINI AI RESPONSE FUNCTION =====
def get_gemini_response(user_input, context=""):
    """Real AI response using Google Gemini"""
    try:
        # Check if Gemini is configured
        if not st.session_state.gemini_key and 'GEMINI_API_KEY' not in st.secrets:
            return "⚠️ Please configure your Google Gemini API key in the sidebar to use the AI chatbot."
        
        # Configure if needed
        if st.session_state.gemini_key:
            genai.configure(api_key=st.session_state.gemini_key)
        
        model = genai.GenerativeModel('gemini-pro')
        
        # Create prompt with context
        prompt = f"""You are an AI assistant for a Text Summarizer app called "Audio to Text Summarizer Using NLP".
The app can summarize audio, video, PDF, URLs, and text using NLP techniques.

Current user context:
- They have {'a summary generated' if context else 'not generated a summary yet'}
- Summary content: {context[:500] if context else 'No summary yet'}

The app features include:
- Summarization using LexRank algorithm
- Keyword extraction
- Word cloud visualization
- QR code generation for sharing
- Translation to Indian languages (Telugu, Hindi, Tamil, etc.)
- Text-to-speech audio download
- History and favorites

User question: {user_input}

Provide a helpful, friendly, and concise answer. If the question is about the app's features, explain them.
If it's about the content, answer based on the context. Be conversational and use emojis occasionally.
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}. Please check your Gemini API key."

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
            response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers=headers,
                data=f
            )
        upload_url = response.json()['upload_url']
        
        transcript_request = {
            'audio_url': upload_url,
            'language_detection': True,
            'speech_models': ['universal-2']
        }
        
        response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            json=transcript_request,
            headers=headers
        )
        transcript_id = response.json()['id']
        
        progress = st.progress(0)
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            
            response = requests.get(
                f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                headers=headers
            )
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
        if not text:
            return None
        
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except:
        return None

# ===== WORD CLOUD GENERATION =====
def create_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            colormap='plasma').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except:
        return None

# ===== QR CODE GENERATION =====
def generate_qr(text):
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(text[:200])
        qr.make(fit=True)
        img = qr.make_image(fill_color="#ff6b6b", back_color="white")
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except:
        return None

# ===== TRANSLATION =====
def translate_summary(text, dest='te'):
    try:
        translator = GoogleTranslator(source='auto', target=dest)
        result = translator.translate(text[:1000])
        return result
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

# ===== DISPLAY RESULTS =====
def display_results(text, source_name):
    if not text or len(text.strip()) == 0:
        st.error("No text to display")
        return
    
    # Store text in session state
    st.session_state.current_text = text
    
    # Calculate statistics
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
            num_points = st.slider(
                "📊 Number of summary sentences:",
                min_value=3,
                max_value=max_val,
                value=st.session_state.slider_value,
                key="main_slider"
            )
            st.session_state.slider_value = num_points
    
    with col2:
        st.metric("Total", total_sentences)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate summary
    if total_sentences <= num_points:
        summary = text
        summary_words = original_words
        st.info("ℹ️ Text has only {} sentences. Showing full text.".format(total_sentences))
    else:
        summary = generate_summary(text, num_points)
        summary_words = len(summary.split())
    
    # Store summary
    st.session_state.current_summary = summary
    
    # Calculate reduction
    if original_words > 0 and summary_words < original_words:
        reduction = int((1 - summary_words/original_words) * 100)
    else:
        reduction = 0
    
    # Add to history
    st.session_state.history.append({
        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'source': source_name,
        'summary': summary[:100] + "...",
        'words': original_words,
        'full_summary': summary
    })
    
    # Display summary
    st.markdown("## 📝 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Characters", f"{original_chars:,}")
    with col2:
        st.metric("📈 Words", f"{original_words:,}")
    with col3:
        st.metric("🔤 Sentences", f"{total_sentences:,}")
    with col4:
        st.metric("📉 Reduced", f"{reduction}%")
    
    # Reading Time
    minutes = original_words // 200
    seconds = int((original_words % 200) / 200 * 60)
    st.metric("⏳ Reading Time", f"{minutes} min {seconds} sec")
    
    # ===== ADVANCED FEATURES =====
    st.markdown("### 🚀 Advanced Features")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("☁️ Word Cloud", key="wc_btn"):
            with st.spinner("Creating word cloud..."):
                fig = create_wordcloud(text)
                if fig:
                    st.session_state.wordcloud_fig = fig
                    st.session_state.show_wordcloud = True
    
    with col2:
        if st.button("📱 QR Code", key="qr_btn"):
            with st.spinner("Generating QR code..."):
                qr_img = generate_qr(summary)
                if qr_img:
                    st.session_state.qr_code = qr_img
                    st.session_state.show_qr = True
    
    with col3:
        if st.button("🌐 Translate", key="trans_btn"):
            with st.spinner("Translating..."):
                translated = translate_summary(summary, 'te')
                if translated:
                    st.session_state.translated_text = translated
                    st.session_state.show_translation = True
    
    with col4:
        if st.button("⭐ Add Favorite", key="fav_btn"):
            st.session_state.favorites.append({
                'title': f"Summary {len(st.session_state.favorites) + 1}",
                'summary': summary,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("✅ Added to favorites!")
    
    with col5:
        if st.button("🔄 Compare", key="comp_btn"):
            st.session_state.show_comparison = True
    
    # Show generated content
    if st.session_state.get('show_wordcloud') and st.session_state.get('wordcloud_fig'):
        st.pyplot(st.session_state.wordcloud_fig)
        if st.button("❌ Close Word Cloud", key="close_wc"):
            st.session_state.show_wordcloud = False
            st.rerun()
    
    if st.session_state.get('show_qr') and st.session_state.get('qr_code'):
        st.image(st.session_state.qr_code, caption="Scan to share summary", width=200)
        if st.button("❌ Close QR", key="close_qr"):
            st.session_state.show_qr = False
            st.rerun()
    
    if st.session_state.get('show_translation') and st.session_state.get('translated_text'):
        st.success("**Telugu Translation:**")
        st.info(st.session_state.translated_text)
        if st.button("❌ Close Translation", key="close_trans"):
            st.session_state.show_translation = False
            st.rerun()
    
    if st.session_state.get('show_comparison'):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔹 Short (3 sentences)**")
            short_summary = generate_summary(text, 3)
            st.info(short_summary[:200] + "..." if len(short_summary) > 200 else short_summary)
        with col2:
            st.markdown("**🔸 Long (7 sentences)**")
            long_summary = generate_summary(text, 7)
            st.info(long_summary[:200] + "..." if len(long_summary) > 200 else long_summary)
        if st.button("❌ Close Comparison", key="close_comp"):
            st.session_state.show_comparison = False
            st.rerun()
    
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

# ===== AI CHATBOT SECTION =====
def display_chatbot():
    st.markdown("### 🤖 AI Assistant (Powered by Google Gemini)")
    
    # Gemini key status
    if not st.session_state.gemini_key and 'GEMINI_API_KEY' not in st.secrets:
        st.warning("⚠️ Please add your Google Gemini API key in the sidebar to use the AI chatbot.")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"<div class='user-message'>👤 {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>🤖 {message['content']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="Ask me anything about your summary or the app...", key="chat_input")
    with col2:
        send_button = st.button("📤 Send", key="send_btn")
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Get AI response
        context = st.session_state.get('current_summary', '')
        with st.spinner("🤔 Thinking..."):
            bot_response = get_gemini_response(user_input, context)
        
        # Add bot response
        st.session_state.chat_history.append({'role': 'bot', 'content': bot_response})
        
        # Clear input and rerun
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# ===== MAIN UI =====
def main():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "🤖 AI Chat", "ℹ️ Help"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a', 'pdf', 'txt']
        )
        
        if uploaded_file:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 {uploaded_file.name} | {file_size:.2f} MB")
            
            if file_ext in ['mp4', 'avi', 'mov']:
                st.video(uploaded_file)
            elif file_ext in ['mp3', 'wav', 'm4a']:
                st.audio(uploaded_file)
            
            if st.button("🚀 Process", key="proc_file"):
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        path = tmp.name
                    
                    if file_ext == 'pdf':
                        text = extract_pdf_text(path)
                        if text:
                            st.success("✅ Extracted PDF")
                            display_results(text, "pdf")
                    elif file_ext == 'txt':
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        display_results(text, "text")
                    else:
                        if not st.session_state.assemblyai_key:
                            st.error("❌ AssemblyAI Key required")
                        else:
                            text = transcribe_with_assemblyai(path)
                            if text:
                                st.success(f"✅ Transcribed: {len(text)} chars")
                                display_results(text, "media")
                    
                    os.unlink(path)
    
    with tab2:
        url = st.text_input("Enter URL", placeholder="https://...")
        
        if url and st.button("🌐 Fetch", key="fetch_url"):
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("Fetching YouTube..."):
                    text, title, content_type = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title}")
                        if content_type == "description":
                            st.info("ℹ️ Showing video description (no captions available)")
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
                <li><strong>
