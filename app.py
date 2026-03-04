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
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    .slider-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>🚀 Advanced Text Summarizer Using NLP</h1><p>Video | Audio | PDF | Text | URL | YouTube | AI Features</p></div>", unsafe_allow_html=True)

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
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = None
if 'qr_code' not in st.session_state:
    st.session_state.qr_code = None
if 'wordcloud_fig' not in st.session_state:
    st.session_state.wordcloud_fig = None

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

# ===== WORD CLOUD GENERATION (No Refresh) =====
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

# ===== QR CODE GENERATION (No Refresh) =====
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

# ===== TRANSLATION (No Refresh) =====
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
    
    summary = f"📌 **MAIN POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
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
    if total_sentences <= num_points:
        summary = text
        summary_words = original_words
        st.info(f"ℹ️ Text has only {total_sentences} sentences. Showing full text.")
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
    st.markdown("## 📋 Summary")
    st.markdown(f"<div class='section-card'>{summary}</div>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Characters", f"{original_chars:,}")
    with col2:
        st.metric("Words", f"{original_words:,}")
    with col3:
        st.metric("Sentences", f"{total_sentences:,}")
    with col4:
        st.metric("Reduced", f"{reduction}%")
    
    # Reading Time
    minutes = original_words // 200
    seconds = int((original_words % 200) / 200 * 60)
    st.metric("⏱️ Reading Time", f"{minutes} min {seconds} sec")
    
    # Advanced Features Section
    st.markdown("### 🚀 Advanced Features")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("☁️ Word Cloud", key="wc_btn"):
            with st.spinner("Creating word cloud..."):
                fig = create_wordcloud(text)
                if fig:
                    st.session_state.wordcloud_fig = fig
    
    with col2:
        if st.button("🔗 QR Code", key="qr_btn"):
            with st.spinner("Generating QR code..."):
                qr_img = generate_qr(summary)
                if qr_img:
                    st.session_state.qr_code = qr_img
    
    with col3:
        if st.button("🌐 Translate", key="trans_btn"):
            with st.spinner("Translating..."):
                translated = translate_summary(summary, 'te')
                if translated:
                    st.session_state.translated_text = translated
    
    with col4:
        title = st.text_input("Title", value="My Summary", key="fav_title", label_visibility="collapsed")
        if st.button("⭐ Add Favorite", key="fav_btn"):
            st.session_state.favorites.append({
                'title': title,
                'summary': summary,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("✅ Added to favorites!")
    
    with col5:
        if st.button("📊 Compare", key="comp_btn"):
            st.session_state.show_comparison = True
    
    # Display generated content
    if st.session_state.get('wordcloud_fig'):
        st.pyplot(st.session_state.wordcloud_fig)
        if st.button("Clear Word Cloud", key="clear_wc"):
            st.session_state.wordcloud_fig = None
            st.rerun()
    
    if st.session_state.get('qr_code'):
        st.image(st.session_state.qr_code, caption="Scan to share summary", width=200)
        if st.button("Clear QR", key="clear_qr"):
            st.session_state.qr_code = None
            st.rerun()
    
    if st.session_state.get('translated_text'):
        st.success(f"**Telugu Translation:**")
        st.info(st.session_state.translated_text)
        if st.button("Clear Translation", key="clear_trans"):
            st.session_state.translated_text = None
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
        if st.button("Close Comparison", key="close_comp"):
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
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li>Get AssemblyAI Key from assemblyai.com</li>
                <li>Upload file or paste URL</li>
                <li>Adjust slider for summary length</li>
                <li>Try advanced features - they won't refresh the page!</li>
                <li>Download text/summary/audio</li>
            </ol>
            
            <h3>✨ Features</h3>
            <ul>
                <li>✅ <strong>No Refresh</strong> - Buttons won't reload page</li>
                <li>✅ <strong>Word Cloud</strong> - Visual keywords</li>
                <li>✅ <strong>QR Code</strong> - Share on mobile</li>
                <li>✅ <strong>Translation</strong> - Telugu support</li>
                <li>✅ <strong>Favorites</strong> - Save summaries</li>
                <li>✅ <strong>Comparison</strong> - 3 vs 7 sentences</li>
                <li>✅ <strong>History</strong> - Recent summaries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
