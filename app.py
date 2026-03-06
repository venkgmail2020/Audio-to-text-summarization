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
import random

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
    .file-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div {
        background-color: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>📁 Multi-File Summarizer</h1><p>Upload multiple files - Process together</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'assemblyai_key' not in st.session_state:
    st.session_state.assemblyai_key = ''
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = {}

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔐 API Configuration")
    assembly_key = st.text_input(
        "🗝️ AssemblyAI Key",
        type="password",
        value=st.session_state.assemblyai_key
    )
    if st.button("💾 Save Keys"):
        st.session_state.assemblyai_key = assembly_key
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if st.session_state.processed_files:
        st.metric("Files Processed", len(st.session_state.processed_files))

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

# ===== TEXT TO SPEECH =====
def text_to_speech(text):
    try:
        text_for_audio = text[:1000] if len(text) > 1000 else text
        tts = gTTS(text=text_for_audio, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except:
        return None

# ===== PROCESS SINGLE FILE =====
def process_single_file(file, file_ext, progress_bar, status_text, file_num, total_files):
    """Process one file with progress tracking"""
    
    status_text.text(f"📁 Processing file {file_num}/{total_files}: {file.name}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    
    if file_ext == 'pdf':
        text = extract_pdf_text(path)
        if text:
            summary = generate_summary(text, 5)
            result = {
                'name': file.name,
                'text': text,
                'summary': summary,
                'status': 'success'
            }
        else:
            result = {'name': file.name, 'status': 'error', 'error': 'Could not extract text'}
    
    elif file_ext == 'txt':
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        summary = generate_summary(text, 5)
        result = {
            'name': file.name,
            'text': text,
            'summary': summary,
            'status': 'success'
        }
    
    else:  # Audio/Video
        if not st.session_state.assemblyai_key:
            result = {'name': file.name, 'status': 'error', 'error': 'AssemblyAI key required'}
        else:
            # Simulate transcription progress
            for i in range(10):
                time.sleep(0.3)
                progress = (file_num - 1 + (i+1)/10) / total_files
                progress_bar.progress(progress)
                status_text.text(f"🎤 Transcribing {file.name}... {i*10}%")
            
            # This would be actual transcription
            text = f"Sample transcription for {file.name}"
            summary = generate_summary(text, 5)
            result = {
                'name': file.name,
                'text': text,
                'summary': summary,
                'status': 'success'
            }
    
    os.unlink(path)
    return result

# ===== MAIN UI =====
def main():
    # ===== MULTIPLE FILE UPLOAD =====
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
                    <b>{file.name}</b> | {size:.2f} MB | Type: {file.type}
                </div>
                """, unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Process All Files", type="primary"):
            
            # ===== PROGRESS BAR SECTION =====
            st.markdown("### 📊 Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            
            # Initialize results
            results = []
            start_time = time.time()
            
            # Process each file
            for idx, file in enumerate(uploaded_files, 1):
                file_ext = file.name.split('.')[-1].lower()
                
                # Process single file
                result = process_single_file(
                    file, file_ext, 
                    progress_bar, status_text, 
                    idx, len(uploaded_files)
                )
                results.append(result)
                
                # Update progress
                progress = idx / len(uploaded_files)
                progress_bar.progress(progress)
                
                # Show time remaining
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = avg_time * (len(uploaded_files) - idx)
                time_text.text(f"⏱️ Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ All files processed!")
            time_text.text(f"⏱️ Total time: {time.time() - start_time:.1f}s")
            
            # Store results
            st.session_state.processed_files = results
            
            # Show summary of results
            st.success(f"✅ Successfully processed {len(results)} files!")
    
    # ===== DISPLAY RESULTS =====
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📝 Summary**")
                        st.info(result['summary'])
                    with col2:
                        words = len(result['text'].split())
                        st.metric("Words", words)
                        st.metric("Summary Words", len(result['summary'].split()))
                        st.metric("Reduction", f"{int((1 - len(result['summary'].split())/words)*100)}%")
                    
                    # Downloads
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

if __name__ == "__main__":
    main()
