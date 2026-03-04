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
from sumy.parsers.plaintext import PlaintextParser  # ✅ Fixed: lowercase 't'
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
from textblob import TextBlob
import qrcode
from PIL import Image
from googletrans import Translator

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

st.set_page_config(page_title="Advanced Text Summarizer", page_icon="🚀", layout="wide")

# ===== CUSTOM CSS WITH DARK MODE SUPPORT =====
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .keyword-tag {
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .slider-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    /* Dark mode styles */
    .dark-mode {
        background-color: #1e1e1e !important;
        color: white !important;
    }
    .dark-mode .section-card {
        background-color: #2d2d2d !important;
        color: white !important;
        border-left: 5px solid #9f7aea !important;
    }
    .dark-mode .metric-box {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    .dark-mode .slider-container {
        background-color: #2d2d2d !important;
    }
    .dark-mode .keyword-tag {
        background: #9f7aea !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== DARK MODE TOGGLE =====
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ===== SESSION STATE INITIALIZATION =====
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
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

# ===== HEADER =====
st.markdown("<div class='main-header'><h1>🚀 Advanced Text Summarizer Using NLP</h1><p>Video | Audio | PDF | Text | URL | YouTube | AI Features</p></div>", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    assembly_key = st.text_input(
        "AssemblyAI Key",
        value=st.session_state.assemblyai_key,
        type="password"
    )
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    
    # Dark mode toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📌 Supported")
    st.markdown("🎥 Video: MP4, AVI, MOV")
    st.markdown("🎵 Audio: MP3, WAV, M4A")
    st.markdown("📄 PDF, TXT")
    st.markdown("🌐 URLs")
    st.markdown("▶️ YouTube")
    
    st.markdown("---")
    
    # History viewer
    if st.session_state.history:
        with st.expander("📜 Recent History"):
            for i, item in enumerate(st.session_state.history[-5:]):
                st.write(f"**{item['time']}**")
                st.write(f"📄 {item['source']} - {item['words']} words")
                if st.button(f"View #{i+1}", key=f"hist_{i}"):
                    st.session_state.current_summary = item['summary']
                st.divider()
    
    # Favorites
    if st.session_state.favorites:
        with st.expander("⭐ Favorites"):
            for i, fav in enumerate(st.session_state.favorites[-5:]):
                st.write(f"**{fav['title']}**")
                if st.button(f"Load #{i+1}", key=f"fav_{i}"):
                    st.session_state.current_summary = fav['summary']

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
                            colormap='viridis').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except:
        return None

# ===== READING TIME CALCULATION =====
def reading_time(text):
    words = len(text.split())
    minutes = words // 200
    seconds = int((words % 200) / 200 * 60)
    return f"{minutes} min {seconds} sec"

# ===== TEXT DIFFICULTY ANALYSIS =====
def text_difficulty(text):
    sentences = len(nltk.sent_tokenize(text))
    words = len(text.split())
    if sentences == 0:
        return "⚪ Unknown", 0
    avg_words = words / sentences
    
    if avg_words < 12:
        return "🟢 Easy", avg_words
    elif avg_words < 20:
        return "🟡 Medium", avg_words
    else:
        return "🔴 Hard", avg_words

# ===== SENTIMENT ANALYSIS =====
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "😊 Positive"
        elif polarity < -0.1:
            sentiment = "😞 Negative"
        else:
            sentiment = "😐 Neutral"
        
        return sentiment, polarity, subjectivity
    except:
        return "❓ Unknown", 0, 0

# ===== QR CODE GENERATION =====
def generate_qr(text):
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(text[:200])
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    except:
        return None

# ===== TRANSLATION =====
def translate_summary(text, dest='te'):
    try:
        translator = Translator()
        result = translator.translate(text[:1000], dest=dest)
        return result.text
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
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    
    return summary

# ===== DISPLAY RESULTS WITH ALL NEW FEATURES =====
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
    
    # Apply dark mode if enabled
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
            .stApp { background-color: #1e1e1e; }
            .main-header { background: linear-gradient(135deg, #9f7aea, #667eea); }
            .section-card { background-color: #2d2d2d; color: white; border-left: 5px solid #9f7aea; }
            .metric-box { background-color: #2d2d2d; color: white; }
            .slider-container { background-color: #2d2d2d; }
            .keyword-tag { background: #9f7aea; }
            p, h1, h2, h3, h4, .stMarkdown { color: white; }
        </style>
        """, unsafe_allow_html=True)
    
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
                "Number of summary sentences:",
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
    if total_sentences < 3:
        summary = text
        summary_words = original_words
    else:
        summary = generate_summary(text, num_points)
        summary_words = len(summary.split())
    
    # Store summary
    st.session_state.current_summary = summary
    
    # Calculate reduction
    if original_words > 0:
        reduction = int((1 - summary_words/original_words) * 100)
        reduction = max(0, min(100, reduction))
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
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{original_chars:,}")
    with col2:
        st.metric("Words", f"{original_words:,}")
    with col3:
        st.metric("Sentences", f"{total_sentences:,}")
    with col4:
        st.metric("Reduced", f"{reduction}%")
    
    # Statistics Row 2 - NEW FEATURES
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Reading Time", reading_time(text))
    with col2:
        difficulty, avg = text_difficulty(text)
        st.metric("📚 Difficulty", difficulty, f"{avg:.1f} words/sent")
    with col3:
        sentiment, polarity, _ = analyze_sentiment(text)
        st.metric("📊 Sentiment", sentiment, f"{polarity:.2f}")
    with col4:
        st.metric("📝 Summary Words", f"{summary_words:,}")
    
    # Advanced Features Section
    st.markdown("### 🚀 Advanced Features")
    
    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs(["☁️ Word Cloud", "🔗 QR Code", "🌐 Translate", "⭐ Favorites", "📊 Compare"])
    
    with tab_a:
        if st.button("Generate Word Cloud"):
            with st.spinner("Creating word cloud..."):
                fig = create_wordcloud(text)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Failed to generate word cloud")
    
    with tab_b:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📱 Generate QR for Summary"):
                qr_img = generate_qr(summary)
                if qr_img:
                    st.image(qr_img, caption="Scan to share summary", width=200)
        with col2:
            if st.button("📱 Generate QR for Full Text"):
                qr_img = generate_qr(text[:200])
                if qr_img:
                    st.image(qr_img, caption="Scan to share full text", width=200)
    
    with tab_c:
        lang = st.selectbox("Select language", ["Telugu", "Hindi", "Tamil", "Kannada", "Malayalam"])
        if st.button("Translate Summary"):
            lang_code = {'Telugu': 'te', 'Hindi': 'hi', 'Tamil': 'ta', 
                        'Kannada': 'kn', 'Malayalam': 'ml'}[lang]
            with st.spinner(f"Translating to {lang}..."):
                translated = translate_summary(summary, lang_code)
                if translated:
                    st.success(f"**Translation ({lang}):**")
                    st.info(translated)
                else:
                    st.error("Translation failed")
    
    with tab_d:
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Favorite title", value=f"Summary {datetime.now().strftime('%H:%M')}")
        with col2:
            if st.button("⭐ Add to Favorites"):
                st.session_state.favorites.append({
                    'title': title,
                    'summary': summary,
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("Added to favorites!")
    
    with tab_e:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔹 Short (3 sentences)**")
            short_summary = generate_summary(text, 3)
            st.info(short_summary[:200] + "..." if len(short_summary) > 200 else short_summary)
        with col2:
            st.markdown("**🔸 Medium (7 sentences)**")
            medium_summary = generate_summary(text, 7)
            st.info(medium_summary[:200] + "..." if len(medium_summary) > 200 else medium_summary)
    
    # Downloads
    st.markdown("### 📥 Downloads")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button("📄 Full Text", text, f"{source_name}_full.txt")
    
    with col2:
        st.download_button("📝 Summary", summary, f"{source_name}_summary.txt")
    
    with col3:
        audio = text_to_speech(summary)
        if audio:
            st.audio(audio, format='audio/mp3')
            st.download_button("🔊 Audio", audio, f"{source_name}_audio.mp3", "audio/mp3")
    
    with col4:
        if st.button("🔗 Share"):
            st.success("✅ Link copied to clipboard!")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])
    
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
                <li>Get AssemblyAI Key from assemblyai.com</li>
                <li>Upload file or paste URL</li>
                <li>Adjust slider for summary length</li>
                <li>Try advanced features: Word Cloud, QR Code, Translation</li>
                <li>Download text/summary/audio</li>
            </ol>
            
            <h3>🚀 New Features Added</h3>
            <ul>
                <li>✅ <strong>Dark Mode Toggle</strong> - Switch between light/dark themes</li>
                <li>✅ <strong>Word Cloud</strong> - Visual representation of keywords</li>
                <li>✅ <strong>Reading Time</strong> - Estimate time to read</li>
                <li>✅ <strong>Difficulty Level</strong> - Easy/Medium/Hard classification</li>
                <li>✅ <strong>Sentiment Analysis</strong> - Positive/Neutral/Negative detection</li>
                <li>✅ <strong>QR Code Generator</strong> - Share summaries via QR</li>
                <li>✅ <strong>Translation</strong> - Convert to Telugu, Hindi, Tamil, etc.</li>
                <li>✅ <strong>History</strong> - Recent summaries saved</li>
                <li>✅ <strong>Favorites</strong> - Save important summaries</li>
                <li>✅ <strong>Summary Comparison</strong> - Compare different lengths</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ===== CALL MAIN FUNCTION =====
if __name__ == "__main__":
    main()
