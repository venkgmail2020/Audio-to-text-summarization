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

st.set_page_config(page_title="Multi-File Summarizer", page_icon="📁", layout="wide")

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #556270);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    .keyword-tag {
        background: linear-gradient(135deg, #ff6b6b, #556270);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .file-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ff6b6b;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div {
        background-color: #ff6b6b !important;
    }
    .current-affairs {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>📁 Multi-File Summarizer</h1><p>Upload multiple files - Get summaries, keywords, audio & current affairs</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔐 API Configuration")
    
    assembly_key = st.text_input(
        "🗝️ AssemblyAI Key (For Video/Audio)",
        value=st.session_state.assemblyai_key,
        type="password",
        placeholder="Enter your AssemblyAI key"
    )
    
    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        
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
        return None, "No content found"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===== YOUTUBE EXTRACTION =====
def extract_youtube_content(url):
    try:
        video_id = None
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('watch?v=')[-1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[-1].split('?')[0]
        if not video_id:
            return None, None, "Invalid YouTube URL"
        
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
                    return full_text, title, "Captions"
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
                return description, title, "Description"
        return None, None, "No captions or description"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

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
        status = st.empty()
        
        for i in range(60):
            time.sleep(2)
            progress.progress(min(i * 2, 95))
            status.text(f"⏳ Transcribing... {i*2}s")
            
            response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
            result = response.json()
            
            if result['status'] == 'completed':
                progress.progress(100)
                status.text("✅ Complete!")
                return result.get('text', '')
            elif result['status'] == 'error':
                return None
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
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
    
    summary = f"📌 **KEY POINTS ({num_points} of {len(sentences)} sentences)**\n\n"
    for i, idx in enumerate(top_indices, 1):
        summary += f"{i}. {sentences[idx]}\n\n"
    return summary

# ===== EXTRACT KEYWORDS =====
def extract_keywords(text, num=15):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return Counter(filtered).most_common(num)

# ===== FORMAT AS CURRENT AFFAIRS =====
def format_as_current_affairs(text, source_name):
    sentences = nltk.sent_tokenize(text)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = Counter(words).most_common(10)
    
    current_affairs = f"""
<div class='current-affairs'>
    <h3>🌍 TODAY'S CURRENT AFFAIRS</h3>
    <p>📅 {datetime.now().strftime("%B %d, %Y")} | 📰 Source: {source_name}</p>
    <hr>
"""
    
    # Main headlines (first 5 sentences)
    current_affairs += "<h4>📰 HEADLINES</h4><ul>"
    for i, sent in enumerate(sentences[:5]):
        if len(sent) > 30:
            current_affairs += f"<li>{sent[:150]}...</li>"
    current_affairs += "</ul>"
    
    # Key terms
    current_affairs += "<h4>🔑 KEY TERMS</h4><p>"
    for word, count in keywords[:8]:
        current_affairs += f"<span class='keyword-tag'>#{word}</span> "
    current_affairs += "</p>"
    
    # Stats
    current_affairs += f"""
    <h4>📊 QUICK STATS</h4>
    <ul>
        <li>Total Sentences: {len(sentences)}</li>
        <li>Reading Time: {len(text.split())//200} min</li>
    </ul>
</div>
"""
    return current_affairs

# ===== PROCESS SINGLE FILE =====
def process_single_file(file, file_ext, progress_bar, status_text, file_num, total_files):
    """Process one file with progress tracking"""
    
    status_text.text(f"📁 Processing file {file_num}/{total_files}: {file.name}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    
    result = {'name': file.name, 'status': 'error', 'error': 'Unknown error'}
    
    if file_ext == 'pdf':
        text = extract_pdf_text(path)
        if text:
            summary = generate_summary(text, 5)
            keywords = extract_keywords(text)
            result = {
                'name': file.name,
                'text': text,
                'summary': summary,
                'keywords': keywords,
                'status': 'success'
            }
        else:
            result = {'name': file.name, 'status': 'error', 'error': 'Could not extract text'}
    
    elif file_ext == 'txt':
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        summary = generate_summary(text, 5)
        keywords = extract_keywords(text)
        result = {
            'name': file.name,
            'text': text,
            'summary': summary,
            'keywords': keywords,
            'status': 'success'
        }
    
    else:  # Audio/Video
        if not st.session_state.assemblyai_key:
            result = {'name': file.name, 'status': 'error', 'error': 'AssemblyAI key required'}
        else:
            for i in range(5):
                time.sleep(0.5)
                progress = (file_num - 1 + (i+1)/5) / total_files
                progress_bar.progress(progress)
                status_text.text(f"🎤 Transcribing {file.name}... {i*20}%")
            
            text = transcribe_with_assemblyai(path)
            if text:
                summary = generate_summary(text, 5)
                keywords = extract_keywords(text)
                result = {
                    'name': file.name,
                    'text': text,
                    'summary': summary,
                    'keywords': keywords,
                    'status': 'success'
                }
            else:
                result = {'name': file.name, 'status': 'error', 'error': 'Transcription failed'}
    
    os.unlink(path)
    return result

# ===== MAIN UI =====
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🌐 URL/YouTube", "📝 Paste Text", "ℹ️ Help"])
    
    with tab1:
        st.markdown("### 📤 Upload Multiple Files")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, TXT, MP4, MP3, etc.)",
            type=['pdf', 'txt', 'mp4', 'mp3', 'wav', 'avi', 'mov'],
            accept_multiple_files=True,
            key="multi_upload"
        )
        
        if uploaded_files:
            st.markdown(f"📊 **Total files:** {len(uploaded_files)}")
            
            # Show file list
            with st.expander("📋 Selected Files", expanded=True):
                for file in uploaded_files:
                    size = len(file.getvalue()) / (1024 * 1024)
                    st.markdown(f"""
                    <div class='file-card'>
                        <b>{file.name}</b> | {size:.2f} MB
                    </div>
                    """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                
                # Progress section
                st.markdown("### 📊 Processing Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                results = []
                start_time = time.time()
                
                for idx, file in enumerate(uploaded_files, 1):
                    file_ext = file.name.split('.')[-1].lower()
                    
                    result = process_single_file(
                        file, file_ext, 
                        progress_bar, status_text, 
                        idx, len(uploaded_files)
                    )
                    results.append(result)
                    
                    progress = idx / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    remaining = (elapsed / idx) * (len(uploaded_files) - idx)
                    time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
                
                progress_bar.progress(1.0)
                status_text.text("✅ All files processed!")
                st.session_state.processed_files = results
                st.success(f"✅ Successfully processed {len(results)} files!")
        
        # Display results
        if st.session_state.processed_files:
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            # Statistics
            success = sum(1 for r in st.session_state.processed_files if r['status'] == 'success')
            failed = len(st.session_state.processed_files) - success
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(st.session_state.processed_files))
            with col2:
                st.metric("✅ Success", success)
            with col3:
                st.metric("❌ Failed", failed)
            
            # Show each file's result
            for idx, result in enumerate(st.session_state.processed_files):
                with st.expander(f"📄 {result['name']}", expanded=(idx==0)):
                    if result['status'] == 'success':
                        # Normal Summary
                        st.markdown("### 📝 Summary")
                        st.info(result['summary'])
                        
                        # Current Affairs Format
                        st.markdown("### 🌍 Current Affairs Format")
                        current_affairs = format_as_current_affairs(result['text'], result['name'])
                        st.markdown(current_affairs, unsafe_allow_html=True)
                        
                        # Statistics
                        words = len(result['text'].split())
                        sentences = len(nltk.sent_tokenize(result['text']))
                        summary_words = len(result['summary'].split())
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📊 Words", words)
                        with col2:
                            st.metric("🔤 Sentences", sentences)
                        with col3:
                            st.metric("📝 Summary Words", summary_words)
                        with col4:
                            reduction = int((1 - summary_words/words) * 100)
                            st.metric("📉 Reduced", f"{reduction}%")
                        
                        # Keywords
                        st.markdown("### 🏷️ Keywords")
                        html = "<div>"
                        for word, count in result['keywords'][:12]:
                            size = min(24 + count, 40)
                            html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Downloads
                        st.markdown("### 📥 Downloads")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📄 Full Text",
                                result['text'],
                                file_name=f"{result['name']}_full.txt"
                            )
                        with col2:
                            st.download_button(
                                "📝 Summary",
                                result['summary'],
                                file_name=f"{result['name']}_summary.txt"
                            )
                        with col3:
                            audio = text_to_speech(result['summary'])
                            if audio:
                                st.audio(audio)
                                st.download_button(
                                    "🔊 Audio",
                                    audio,
                                    file_name=f"{result['name']}_audio.mp3"
                                )
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.markdown("### 🔗 URL/YouTube")
        url = st.text_input("Enter URL", placeholder="https://example.com/article or YouTube link")
        
        if url and st.button("🌐 Fetch & Summarize", key="fetch_url"):
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner("Fetching YouTube..."):
                    text, title, source = extract_youtube_content(url)
                    if text:
                        st.success(f"✅ {title} ({source})")
                        
                        # Create a temporary result
                        result = {
                            'name': title,
                            'text': text,
                            'summary': generate_summary(text, 5),
                            'keywords': extract_keywords(text),
                            'status': 'success'
                        }
                        
                        # Display
                        st.markdown("### 📝 Summary")
                        st.info(result['summary'])
                        
                        st.markdown("### 🌍 Current Affairs Format")
                        current_affairs = format_as_current_affairs(text, "YouTube")
                        st.markdown(current_affairs, unsafe_allow_html=True)
                        
                        # Keywords
                        st.markdown("### 🏷️ Keywords")
                        html = "<div>"
                        for word, count in result['keywords'][:12]:
                            size = min(24 + count, 40)
                            html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)
                        
                        # Downloads
                        audio = text_to_speech(result['summary'])
                        if audio:
                            st.audio(audio)
                            st.download_button("🔊 Download Audio", audio, "summary.mp3")
                    else:
                        st.warning("No content found")
            elif validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text:
                        st.success(f"✅ {title}")
                        
                        result = {
                            'name': title,
                            'text': text,
                            'summary': generate_summary(text, 5),
                            'keywords': extract_keywords(text),
                            'status': 'success'
                        }
                        
                        st.markdown("### 📝 Summary")
                        st.info(result['summary'])
                        
                        st.markdown("### 🌍 Current Affairs Format")
                        current_affairs = format_as_current_affairs(text, "Web Article")
                        st.markdown(current_affairs, unsafe_allow_html=True)
                        
                        st.markdown("### 🏷️ Keywords")
                        html = "<div>"
                        for word, count in result['keywords'][:12]:
                            size = min(24 + count, 40)
                            html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)
                        
                        audio = text_to_speech(result['summary'])
                        if audio:
                            st.audio(audio)
                            st.download_button("🔊 Download Audio", audio, "summary.mp3")
                    else:
                        st.warning("No content found")
            else:
                st.error("Invalid URL")
    
    with tab3:
        st.markdown("### 📝 Paste Text")
        text_input = st.text_area("Paste your text here", height=200)
        
        if text_input and st.button("📝 Summarize", key="summ_text"):
            if len(text_input) > 200:
                st.markdown("### 📝 Summary")
                summary = generate_summary(text_input, 5)
                st.info(summary)
                
                st.markdown("### 🌍 Current Affairs Format")
                current_affairs = format_as_current_affairs(text_input, "Pasted Text")
                st.markdown(current_affairs, unsafe_allow_html=True)
                
                keywords = extract_keywords(text_input)
                st.markdown("### 🏷️ Keywords")
                html = "<div>"
                for word, count in keywords[:12]:
                    size = min(24 + count, 40)
                    html += f"<span class='keyword-tag' style='font-size: {size}px;'>{word} ({count})</span> "
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
                
                audio = text_to_speech(summary)
                if audio:
                    st.audio(audio)
                    st.download_button("🔊 Download Audio", audio, "summary.mp3")
            else:
                st.warning("Text too short (min 200 chars)")
    
    with tab4:
        st.markdown("""
        <div class='section-card'>
            <h3>📌 How to Use</h3>
            <ol>
                <li><strong>File Upload:</strong> Upload multiple files (PDF, TXT, MP4, MP3)</li>
                <li><strong>URL/YouTube:</strong> Paste any article or video link</li>
                <li><strong>Paste Text:</strong> Direct text analysis</li>
            </ol>
            
            <h3>✨ Features</h3>
            <ul>
                <li>📁 <strong>Multiple File Upload</strong> - Process many files at once</li>
                <li>📝 <strong>Summary</strong> - Extract key points</li>
                <li>🌍 <strong>Current Affairs Format</strong> - View as news headlines</li>
                <li>🏷️ <strong>Keywords</strong> - Important terms highlighted</li>
                <li>🔊 <strong>Audio Download</strong> - Listen to summaries</li>
                <li>📥 <strong>Download</strong> - Text, summary, audio</li>
            </ul>
            
            <h3>📊 Supported Formats</h3>
            <ul>
                <li>🎥 Video: MP4, AVI, MOV</li>
                <li>🎵 Audio: MP3, WAV, M4A</li>
                <li>📄 Document: PDF, TXT</li>
                <li>🌐 Online: URLs, YouTube</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
