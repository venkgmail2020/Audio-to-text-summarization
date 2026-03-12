import streamlit as st
import requests
import nltk
import re
import io
import tempfile
import os
from collections import Counter
from gtts import gTTS
from bs4 import BeautifulSoup
import validators
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import matplotlib.pyplot as plt
import PyPDF2

# ---------- NLTK ----------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# ---------- PAGE ----------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Research Assistant")
st.write("Video | Audio | PDF | URL | Text Summarizer")

# ---------- SESSION ----------
if "text" not in st.session_state:
    st.session_state.text = ""

# ---------- PDF ----------
def extract_pdf(file):

    text = ""

    try:
        pdf = PyPDF2.PdfReader(file)

        for page in pdf.pages:
            content = page.extract_text()

            if content:
                text += content

        return text

    except Exception as e:
        st.error(e)
        return None

# ---------- URL ----------
def extract_url(url):

    try:

        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, headers=headers)

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text() for p in paragraphs])

        return text

    except Exception as e:

        st.error(e)

        return None

# ---------- YOUTUBE ----------
def extract_youtube(url):

    try:

        video_id = None

        if "watch?v=" in url:
            video_id = url.split("watch?v=")[1]

        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1]

        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        text = " ".join([x["text"] for x in transcript])

        return text

    except:

        try:

            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:

                info = ydl.extract_info(url, download=False)

                return info.get("description")

        except Exception as e:

            st.error(e)

            return None

# ---------- SUMMARY ----------
def summarize(text, n=5):

    sentences = nltk.sent_tokenize(text)

    words = re.findall(r"\w+", text.lower())

    stop = set(nltk.corpus.stopwords.words("english"))

    words = [w for w in words if w not in stop]

    freq = Counter(words)

    scores = {}

    for sent in sentences:

        for word in sent.lower().split():

            if word in freq:

                scores[sent] = scores.get(sent, 0) + freq[word]

    top = sorted(scores, key=scores.get, reverse=True)[:n]

    return "\n".join(top)

# ---------- KEYWORDS ----------
def keywords(text):

    words = re.findall(r"\w+", text.lower())

    stop = set(nltk.corpus.stopwords.words("english"))

    words = [w for w in words if w not in stop]

    return Counter(words).most_common(10)

# ---------- TEXT TO SPEECH ----------
def speak(text):

    try:

        tts = gTTS(text=text[:1000], lang="en")

        audio = io.BytesIO()

        tts.write_to_fp(audio)

        audio.seek(0)

        return audio

    except Exception as e:

        st.error(e)

        return None

# ---------- INFOGRAPHIC ----------
def infographic(text):

    words = re.findall(r"\w+", text.lower())

    top = Counter(words).most_common(10)

    labels = [x[0] for x in top]

    values = [x[1] for x in top]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    plt.xticks(rotation=45)

    return fig

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📁 Upload", "🌐 URL", "📝 Text"])

# ---------- FILE ----------
with tab1:

    file = st.file_uploader("Upload PDF / TXT")

    if file:

        if file.type == "application/pdf":

            text = extract_pdf(file)

        else:

            text = file.read().decode()

        st.session_state.text = text

# ---------- URL ----------
with tab2:

    url = st.text_input("Enter URL")

    if st.button("Fetch"):

        if "youtube" in url or "youtu.be" in url:

            text = extract_youtube(url)

        elif validators.url(url):

            text = extract_url(url)

        else:

            st.error("Invalid URL")

            text = None

        st.session_state.text = text

# ---------- TEXT ----------
with tab3:

    text = st.text_area("Paste text")

    if st.button("Process"):

        st.session_state.text = text

# ---------- OUTPUT ----------
if st.session_state.text:

    text = st.session_state.text

    st.subheader("📝 Summary")

    summary = summarize(text)

    st.write(summary)

    st.subheader("🏷️ Keywords")

    for k, v in keywords(text):

        st.write(k, v)

    st.subheader("📊 Infographic")

    fig = infographic(text)

    st.pyplot(fig)

    st.subheader("🔊 Audio")

    audio = speak(summary)

    if audio:

        st.audio(audio)

        st.download_button(
            "Download Audio",
            audio,
            "summary.mp3"
        )
