# 🎬 Movie Review Sentiment Analyzer

**Turn movie opinions into insights in less than 3 seconds.**

This tool classifies movie reviews as **Positive**, **Negative**, or **Neutral** using **Google's Gemini-2.0-Flash** model through sophisticated prompt engineering. Features both single-review analysis and batch processing capabilities with confidence scoring and evidence extraction.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Google AI Studio API key ([Get yours free](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```
   
   Or create a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Launch the application**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   👉 **Access at: [http://localhost:8501](http://localhost:8501)**

---

## 🎭 Features & Usage

### Single Review Analysis
- **Input**: Paste any movie review text
- **Analysis Modes**: 
  - **Normal Mode**: Detects subtle sentiment cues
  - **Strict Mode**: Requires strong, unambiguous sentiment
- **Output**: Label + Confidence score + Explanation + Evidence phrases

### Batch Processing
- **Upload CSV** with `review` column
- **Download results** as CSV with sentiment analysis
- **Progress tracking** for large datasets
- **Error handling** for malformed reviews

### Key Capabilities
- ⚡ **Sub-3 second response time** for single reviews
- 🎯 **Confidence scoring** (0.0 - 1.0 scale)
- 🔍 **Evidence phrases** extraction
- 🛡️ **Robust error handling** with 3-retry mechanism
- 📊 **Batch processing** with rate limiting
- 🎨 **Clean, responsive UI** with custom styling

---

## 🏗️ Architecture

```
├── streamlit_app.py      # Main web interface & batch processing
├── sentiment_llm.py      # Core sentiment analysis logic & prompts  
├── requirements.txt      # Python dependencies
├── test_dataset.csv      # 42-sample balanced test set
├── README.md            # Setup & usage documentation
└── REPORT.md            # Technical analysis & metrics
```

---

## 📊 Technical Specifications

- **Model**: Google Gemini-1.5-Flash (free tier)
- **Temperature**: 0.1 (deterministic outputs)
- **Output Format**: Structured JSON with validation
- **Rate Limiting**: 200ms between batch requests
- **Retry Logic**: 3 attempts with exponential backoff
- **Response Time**: ~2 seconds average

---

## 🧪 Prompt Design

### Core Strategy
**Dual-mode prompting system** with explicit JSON schema enforcement:

**Normal Mode**: Detects subtle emotional cues and implicit sentiment
```
"I found myself surprisingly invested in the character's journey" → Positive
```

**Strict Mode**: Requires clear, unambiguous sentiment markers
```  
"The movie was okay, some good parts" → Neutral (not Positive)
```

### Key Design Elements
- **Few-shot examples** embedded in prompts
- **Explicit JSON schema** with field validation
- **Evidence phrase extraction** for explainability
- **Conservative fallback** to Neutral on ambiguity
- **Temperature control** for consistent outputs

---

## 📋 CSV Format

**Required columns:**
```csv
review
"Amazing film with great acting!"
"Boring and too long."
```

**Optional columns:**
```csv
review,movie_title
"Amazing film with great acting!",Top Gun
"Boring and too long.",The Movie
```

---

## 🛠️ Troubleshooting

### Common Issues

**API Key Errors**
```bash
# Verify your key is set
echo $GEMINI_API_KEY
# Or check .env file exists
```

**Import Errors**  
```bash
pip install -r requirements.txt --upgrade
```

**CSV Processing Issues**
- Ensure `review` column exists
- Check for proper CSV encoding (UTF-8)
- Verify no empty rows

**Slow Response Times**
- Check internet connection
- API quota may be exceeded
- Try reducing batch size

---


## Support

For issues, improvements, or questions:
1. Check the troubleshooting section above
2. Review `REPORT.md` for technical details  
3. Test with the provided `test_dataset.csv`
4. Verify API key configuration and quotas

---