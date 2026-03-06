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

# Try to import Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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

# ===== CUSTOM CSS WITH PROPER CONTRAST =====
st.markdown("""
<style>
    /* Main background - LIGHT THEME (white background, black text) */
    .stApp {
        background: white !important;
    }
    
    /* All text black for readability */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, 
    .stTextInput label, .stSelectbox label, .stTextInput input, 
    .stTextArea textarea, .stMetric label, .stMetric value {
        color: black !important;
    }
    
    /* Input fields - white background, black text */
    .stTextInput > div > div > input {
        background: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #666 !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }
    
    /* Main header - keep gradient but ensure text readable */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        color: white !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }
    .main-header h1, .main-header p {
        color: white !important;
    }
    
    /* Section cards - light background with black text */
    .section-card {
        background: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 5px solid #ff6b6b !important;
        margin: 1rem 0 !important;
        color: black !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Keyword tags - keep colored but ensure text readable */
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
        background: #f8f9fa !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
        color: black !important;
        text-align: center !important;
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
    
    /* Slider */
    .slider-container {
        background: #f0f2f6 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid #ddd !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #f0f2f6 !important;
        padding: 0.5rem !important;
        border-radius: 10px !important;
        gap: 0.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: black !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        color: white !important;
    }
    
    /* Chat container - light background */
    .chat-container {
        background: #f8f9fa !important;
        border-radius: 15px !important;
        padding: 20px !important;
        min-height: 400px !important;
        max-height: 500px !important;
        margin-bottom: 20px !important;
        border: 1px solid #ddd !important;
        overflow-y: auto !important;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #ff6b6b, #556270) !important;
        color: white !important;
        padding: 12px 18px !important;
        border-radius: 20px 20px 5px 20px !important;
        margin: 10px 0 10px auto !important;
        max-width: 80% !important;
        width: fit-content !important;
        clear: both !important;
        float: right !important;
        word-wrap: break-word !important;
    }
    .bot-message {
        background: #e9ecef !important;
        color: black !important;
        padding: 12px 18px !important;
        border-radius: 20px 20px 20px 5px !important;
        margin: 10px auto 10px 0 !important;
        max-width: 80% !important;
        width: fit-content !important;
        clear: both !important;
        float: left !important;
        word-wrap: break-word !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center !important;
        padding: 80px 20px !important;
        color: #666 !important;
    }
    .welcome-message h2 {
        color: #ff6b6b !important;
        font-size: 2.5em !important;
        margin-bottom: 20px !important;
    }
    .welcome-message p {
        color: #666 !important;
        font-size: 1.2em !important;
    }
    
    /* Current Affairs Box */
    .current-affairs {
        background: #f8f9fa !important;
        border-left: 5px solid #ff6b6b !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        margin: 1rem 0 !important;
        color: black !important;
    }
    .current-affairs h3 {
        color: #ff6b6b !important;
        border-bottom: 1px solid #dee2e6 !important;
        padding-bottom: 10px !important;
    }
    .current-affairs ul {
        list-style-type: none !important;
        padding-left: 0 !important;
    }
    .current-affairs li {
        margin: 10px 0 !important;
        padding-left: 20px !important;
        border-left: 3px solid #ff6b6b !important;
        color: black !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    [data-testid="stSidebar"] * {
        color: black !important;
    }
    
    /* Clear floats */
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class='main-header'>
    <h1>🎤 Audio to Text Summarizer Using NLP</h1>
    <p style='font-size: 1.2rem; opacity: 0.9;'>Transform Audio | Video | PDF | Text | URL into Smart Summaries & Current Affairs</p>
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
        type="password",
        placeholder="Enter your AssemblyAI key"
    )
    
    gemini_key = st.text_input(
        "🤖 Google Gemini Key (Optional)",
        value=st.session_state.gemini_key,
        type="password",
        placeholder="Get from aistudio.google.com"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.session_state.gemini_key = gemini_key
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

# ===== FORMAT AS CURRENT AFFAIRS =====
def format_as_current_affairs(text, source_name):
    """Convert extracted text into Current Affairs format"""
    
    sentences = nltk.sent_tokenize(text)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = Counter(words).most_common(15)
    
    current_affairs = f"""
<div class='current-affairs'>
    <h3>🌍 TODAY'S CURRENT AFFAIRS</h3>
    <p style='color: #666;'>📅 {datetime.now().strftime("%B %d, %Y")} | 📰 Source: {source_name}</p>
    <hr style='border-color: #dee2e6;'>
"""
    
    # Main headlines (first 3 important sentences)
    main_headlines = []
    for sent in sentences[:10]:
        if any(word[0] in sent.lower() for word in keywords[:5]):
            main_headlines.append(sent)
            if len(main_headlines) >= 3:
                break
    
    if main_headlines:
        current_affairs += "<h4>📰 MAIN HEADLINES</h4><ul>"
        for headline in main_headlines[:3]:
            current_affairs += f"<li>{headline[:200]}...</li>"
        current_affairs += "</ul>"
    
    # Key terms
    current_affairs += "<h4>🔑 KEY TERMS</h4><p>"
    for word, count in keywords[:10]:
        current_affairs += f"<span class='keyword-tag'>#{word.title()}</span> "
    current_affairs += "</p>"
    
    # Quick stats
    current_affairs += f"""
    <h4>📊 QUICK STATS</h4>
    <ul>
        <li>Total Sentences: {len(sentences)}</li>
        <li>Main Topics: {len(main_headlines)}</li>
        <li>Reading Time: {len(text.split())//200} min</li>
    </ul>
</div>
"""
    return current_affairs

# ===== SIMPLE CHATBOT RESPONSE (Working properly) =====
def get_bot_response(user_input, context=""):
    """Simple but working chatbot response"""
    user_input = user_input.lower().strip()
    
    # Greetings
    if any(word in user_input for word in ['hi', 'hello', 'hey', 'namaste', 'hy']):
        return "Hello! How can I help you with your summary today?"
    
    # Questions about user
    if 'your name' in user_input or 'who are you' in user_input:
        return "I'm your AI assistant for this Text Summarizer app!"
    
    # Questions about summary
    if 'summary' in user_input:
        if context:
            return f"Your current summary: {context[:150]}... You can adjust the length using the slider."
        return "No summary yet. Upload a file or paste text first!"
    
    # Questions about keywords
    if 'keyword' in user_input:
        return "Keywords are important words in your text. They appear as colored tags below the summary."
    
    # Questions about features
    if 'feature' in user_input or 'what can you do':
        return "I can help with summarization, keywords, word cloud, QR codes, translation, and more!"
    
    # Questions about help
    if 'help' in user_input:
        return "Check the Help tab for detailed instructions on how to use all features."
    
    # Questions about audio
    if 'audio' in user_input or 'listen' in user_input:
        return "You can listen to your summary by clicking the Audio download button."
    
    # Questions about translation
    if 'translate' in user_input or 'telugu' in user_input:
        return "You can translate your summary to Telugu, Hindi, and Tamil using the Translate button."
    
    # Questions about current affairs
    if 'current affairs' in user_input or 'news' in user_input:
        return "Your uploaded content is shown in Current Affairs format in the second tab!"
    
    # Default response
    return "I'm not sure I understand. Try asking about summary, keywords, features, or help."

# ===== DISPLAY RESULTS WITH CURRENT AFFAIRS TAB =====
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
                min_value=3,
                max_value=max_val,
                value=st.session_state.slider_value,
                key="main_slider"
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
    
    st.session_state.history.append({
        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'source': source_name,
        'summary': summary[:100] + "...",
        'words': original_words,
        'full_summary': summary
    })
    
    # Create tabs for Summary and Current Affairs
    tab1, tab2 = st.tabs(["📋 Summary", "🌍 Current Affairs Format"])
    
    with tab1:
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
        
        # Advanced Features
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
    
    with tab2:
        st.markdown("## 🌍 Today's Current Affairs")
        current_affairs = format_as_current_affairs(text, source_name)
        st.markdown(current_affairs, unsafe_allow_html=True)
        
        # Download Current Affairs
        plain_text = f"TODAY'S CURRENT AFFAIRS - {datetime.now().strftime('%B %d, %Y')}\n\n"
        plain_text += "MAIN HEADLINES:\n"
        for i, sent in enumerate(nltk.sent_tokenize(text)[:5]):
            plain_text += f"{i+1}. {sent}\n\n"
        
        st.download_button(
            "📥 Download as Current Affairs",
            plain_text,
            file_name=f"current_affairs_{datetime.now().strftime('%Y%m%d')}.txt"
        )

# ===== AI CHATBOT SECTION - FIXED =====
def display_chatbot():
    st.markdown("### 🤖 AI Assistant")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Check if Gemini is available
    has_gemini = GEMINI_AVAILABLE and (st.session_state.gemini_key or 'GEMINI_API_KEY' in st.secrets)
    
    if has_gemini:
        st.success("✅ Gemini AI is connected! Ask me anything.")
    else:
        st.info("💡 Using simple chatbot. Add Gemini API key for AI-powered responses.")
    
    # Chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div class='welcome-message'>
            <h2>👋 Hello!</h2>
            <p>Ask me anything about your summary or app features</p>
            <p style='color: #666; margin-top: 20px;'>Type your message below and click Send</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='user-message'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='clearfix'></div>", unsafe_allow_html=True)
    
    # Chat input - FIXED BRACKET ISSUE
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your message here...", key="chat_input_fixed")  # ✅ Closed properly
    with col2:
        send = st.button("📤 Send", key="send_btn_fixed", use_container_width=True)
    
    if send and user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Get response
        context = st.session_state.get('current_summary', '')
        response = get_bot_response(user_input, context)
        
        # Add bot response
        st.session_state.chat_history.append({'role': 'bot', 'content': response})
        st.rerun()
    
    # Clear button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat", key="clear_chat_fixed"):
        st.session_state.chat_history = []
        st.rerun()
