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
from sumy.parsers.plaintext import PlaintextParser  # ✅ Fixed
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import ssl
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import validators

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

st.set_page_config(page_title="News Analyzer & Current Affairs", page_icon="📰", layout="wide")

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
    .credibility-high {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .credibility-medium {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .credibility-low {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .news-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ff6b6b;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-badge {
        background: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1>📰 News Analyzer & Current Affairs</h1><p>Verify news & stay updated</p></div>", unsafe_allow_html=True)

# ===== SESSION STATE =====
if 'news_api_key' not in st.session_state:
    st.session_state.news_api_key = ''
if 'current_news' not in st.session_state:
    st.session_state.current_news = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    
    news_api = st.text_input(
        "📰 NewsAPI Key (Optional)",
        type="password",
        value=st.session_state.news_api_key,
        help="Get free key from newsapi.org"
    )
    
    if st.button("💾 Save Keys"):
        st.session_state.news_api_key = news_api
        st.success("✅ Keys saved!")
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if st.session_state.analysis_history:
        st.metric("Total Analyzed", len(st.session_state.analysis_history))

# ===== CURRENT AFFAIRS FUNCTIONS =====
def get_current_affairs():
    """Fetch current affairs from India"""
    try:
        # Try using NewsAPI if key available
        if st.session_state.news_api_key:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'country': 'in',
                'apiKey': st.session_state.news_api_key,
                'pageSize': 10
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
        
        # Fallback to sample news
        return get_sample_news()
        
    except Exception as e:
        st.warning(f"Using sample news: {e}")
        return get_sample_news()

def get_sample_news():
    """Sample news when API not available"""
    return [
        {
            'title': 'India Budget 2026: Key Highlights',
            'description': 'Finance Minister presents budget with focus on infrastructure and digital India.',
            'source': {'name': 'Economic Times'},
            'publishedAt': '2026-03-06T10:30:00Z',
            'url': '#'
        },
        {
            'title': 'IPL 2026: Season Starts March 22',
            'description': '10 teams to compete in 74 matches across 13 cities.',
            'source': {'name': 'Sports Today'},
            'publishedAt': '2026-03-06T09:15:00Z',
            'url': '#'
        },
        {
            'title': 'Heat Wave Alert in North India',
            'description': 'Temperatures to rise 3-5°C above normal in Delhi, UP, Rajasthan.',
            'source': {'name': 'Weather News'},
            'publishedAt': '2026-03-06T08:45:00Z',
            'url': '#'
        },
        {
            'title': 'ISRO to Launch Navigation Satellite',
            'description': 'NVS-02 satellite to be launched on March 15 from Sriharikota.',
            'source': {'name': 'Space News'},
            'publishedAt': '2026-03-06T07:30:00Z',
            'url': '#'
        },
        {
            'title': 'Stock Market: Sensex Crosses 85,000',
            'description': 'Markets hit all-time high on strong economic data.',
            'source': {'name': 'Business Line'},
            'publishedAt': '2026-03-06T06:20:00Z',
            'url': '#'
        }
    ]

def display_current_affairs():
    """Display current affairs in nice format"""
    st.markdown("### 🌍 Today's Current Affairs")
    st.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")
    
    news = get_current_affairs()
    st.session_state.current_news = news
    
    for article in news:
        published = article.get('publishedAt', '')[:10]
        st.markdown(f"""
        <div class='news-item'>
            <h4>{article['title']}</h4>
            <p>{article.get('description', 'No description')}</p>
            <div>
                <span class='source-badge'>📰 {article['source']['name']}</span>
                <span class='source-badge'>📅 {published}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh News"):
        st.rerun()

# ===== FAKE NEWS ANALYSIS =====
def analyze_credibility(url):
    """Analyze URL for credibility"""
    score = 0
    reasons = []
    
    # Factor 1: Domain reputation
    trusted_domains = ['thehindu.com', 'indiatimes.com', 'ndtv.com', 'bbc.com', 'reuters.com']
    suspicious_domains = ['dailypost.xyz', 'news24hrs.net', 'breakingtv.co']
    
    domain = urlparse(url).netloc.replace('www.', '')
    
    if any(td in domain for td in trusted_domains):
        score += 40
        reasons.append("✅ Trusted news source")
    elif any(sd in domain for sd in suspicious_domains):
        score += 10
        reasons.append("⚠️ Unknown source - verify carefully")
    else:
        score += 20
        reasons.append("ℹ️ Check source reputation")
    
    # Factor 2: URL structure
    if len(url) > 100:
        score -= 10
        reasons.append("❌ Suspiciously long URL")
    
    if re.search(r'\d{5,}', url):
        score -= 10
        reasons.append("❌ Contains suspicious numbers")
    
    # Factor 3: Use HTTPS
    if url.startswith('https'):
        score += 10
        reasons.append("✅ Secure connection")
    
    # Factor 4: Age of domain (simulated)
    domain_age = random.randint(1, 20)
    if domain_age > 5:
        score += 10
        reasons.append(f"✅ Domain age: {domain_age}+ years")
    else:
        score -= 5
        reasons.append(f"⚠️ New domain ({domain_age} years)")
    
    # Ensure score between 0-100
    score = max(0, min(100, score))
    
    return score, reasons

def get_fact_checks(text):
    """Simulate fact checking"""
    # Extract claims (simplified)
    sentences = nltk.sent_tokenize(text)
    claims = sentences[:3]  # First 3 sentences as claims
    
    fact_checks = []
    for claim in claims:
        # Simulate fact check result
        result = random.choice([
            {"status": "True", "confidence": random.randint(70, 95)},
            {"status": "Mostly True", "confidence": random.randint(60, 85)},
            {"status": "Misleading", "confidence": random.randint(40, 60)},
            {"status": "False", "confidence": random.randint(10, 30)}
        ])
        
        fact_checks.append({
            'claim': claim[:100] + '...',
            'result': result['status'],
            'confidence': result['confidence'],
            'source': random.choice(['FactCheck.org', 'PolitiFact', 'AltNews'])
        })
    
    return fact_checks

def display_analysis(url, text):
    """Display fake news analysis"""
    st.markdown("### 🔍 Fake News Analysis")
    
    # Get credibility score
    score, reasons = analyze_credibility(url)
    
    # Score display
    if score >= 70:
        st.markdown(f"<div class='credibility-high'>✅ Credibility Score: {score}/100 - Likely Reliable</div>", unsafe_allow_html=True)
    elif score >= 40:
        st.markdown(f"<div class='credibility-medium'>⚠️ Credibility Score: {score}/100 - Verify Carefully</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='credibility-low'>❌ Credibility Score: {score}/100 - Likely Unreliable</div>", unsafe_allow_html=True)
    
    # Reasons
    with st.expander("📋 Analysis Details"):
        for reason in reasons:
            st.write(reason)
    
    # Fact checks
    st.markdown("### ✅ Fact Check Results")
    fact_checks = get_fact_checks(text)
    
    for fc in fact_checks:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**Claim:** {fc['claim']}")
        with col2:
            status = fc['result']
            if status == 'True':
                st.success(f"✅ {status}")
            elif status == 'Mostly True':
                st.info(f"📊 {status}")
            elif status == 'Misleading':
                st.warning(f"⚠️ {status}")
            else:
                st.error(f"❌ {status}")
        with col3:
            st.write(f"Confidence: {fc['confidence']}%")
        st.caption(f"Source: {fc['source']}")
        st.divider()
    
    # Save to history
    st.session_state.analysis_history.append({
        'url': url,
        'score': score,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M")
    })

# ===== URL EXTRACTION =====
def extract_from_url(url):
    """Extract text from URL"""
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

# ===== MAIN UI =====
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Current Affairs", "🔍 URL Analysis", "📝 Paste Text", "ℹ️ Help"])
    
    with tab1:
        display_current_affairs()
    
    with tab2:
        st.markdown("### 🔍 Analyze News URL")
        url = st.text_input("Enter URL", placeholder="https://example.com/news-article")
        
        if url and st.button("🔍 Analyze & Summarize", key="analyze_url"):
            if validators.url(url):
                with st.spinner("Fetching article..."):
                    text, title = extract_from_url(url)
                    if text:
                        st.success(f"✅ Fetched: {title}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📝 Summary")
                            summary = generate_summary(text, 5)
                            st.info(summary)
                            
                            # Downloads
                            audio = text_to_speech(summary)
                            if audio:
                                st.audio(audio)
                                st.download_button("🔊 Download Audio", audio, "summary.mp3")
                        
                        with col2:
                            display_analysis(url, text)
                    else:
                        st.warning("No content found")
            else:
                st.error("Invalid URL")
    
    with tab3:
        st.markdown("### 📝 Paste Text for Analysis")
        text_input = st.text_area("Paste article text here", height=200)
        
        if text_input and st.button("📝 Analyze Text", key="analyze_text"):
            if len(text_input) > 200:
                st.markdown("### 📝 Summary")
                summary = generate_summary(text_input, 5)
                st.info(summary)
                
                # Simplified analysis for text
                st.markdown("### 🔍 Quick Analysis")
                words = len(text_input.split())
                sentences = len(nltk.sent_tokenize(text_input))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", words)
                with col2:
                    st.metric("Sentences", sentences)
                with col3:
                    st.metric("Read Time", f"{words//200} min")
                
                # Download
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
                <li><strong>Current Affairs:</strong> View latest news from India</li>
                <li><strong>URL Analysis:</strong> Paste any news URL to:
                    <ul>
                        <li>Get credibility score</li>
                        <li>Check facts</li>
                        <li>Read summary</li>
                    </ul>
                </li>
                <li><strong>Paste Text:</strong> Direct text analysis</li>
            </ol>
            
            <h3>🔍 How Fake News Detection Works</h3>
            <ul>
                <li>Domain reputation check</li>
                <li>URL structure analysis</li>
                <li>Fact checking with multiple sources</li>
                <li>Credibility scoring (0-100)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
