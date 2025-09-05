
# Core imports for sentiment analysis using Google's Gemini AI
import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"  # Fast model for efficient sentiment analysis

def validate_and_clean_result(raw_result):
    """Clean up AI responses to ensure we get reliable, standardized results."""
    # Make sure sentiment label is valid - default to Neutral if weird response
    label = str(raw_result.get("label", "Neutral")).strip().title()
    
    if label not in {"Positive", "Negative", "Neutral"}:
        label = "Neutral"
    
    # Ensure confidence score is a valid number between 0 and 1
    try:
        confidence = float(raw_result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
    except (ValueError, TypeError):
        confidence = 0.5  # Default to neutral confidence
    
    # Get explanation text, provide fallback if missing
    explanation = str(raw_result.get("explanation", "")).strip()
    if not explanation:
        explanation = "No detailed explanation was provided for this classification."
    
    # Clean up evidence phrases that influenced the decision
    evidence = raw_result.get("evidence_phrases", [])
    if not isinstance(evidence, list):
        evidence = []
    
    clean_evidence = []
    for phrase in evidence:
        clean_phrase = str(phrase).strip()[:150]  # Trim overly long phrases
        if clean_phrase:
            clean_evidence.append(clean_phrase)
    
    clean_evidence = clean_evidence[:6]  # Keep only top 6 most relevant phrases
    
    return {
        "label": label,
        "confidence": confidence,
        "explanation": explanation,
        "evidence_phrases": clean_evidence,
    }

def analyze_sentiment(review_text, analysis_mode="lenient"):
    """Main function to analyze sentiment of movie review text.
    
    Args:
        review_text (str): Review text to analyze
        analysis_mode (str): "strict" for conservative analysis, "lenient" for subtle cues
    """
    # Handle edge case: empty or invalid input
    if not isinstance(review_text, str) or not review_text.strip():
        return {
            "label": "Neutral",
            "confidence": 0.5,
            "explanation": "Cannot analyze empty text",
            "evidence_phrases": [],
        }
    
    # Choose prompt based on analysis mode - strict vs lenient
    if analysis_mode == "strict":
        prompt_text = f"""
You are a conservative film critic assistant. Your task is to analyze a movie review for its sentiment, requiring STRONG, UNAMBIGUOUS evidence for a positive or negative classification.

Your response MUST be a valid JSON object in this exact format:
{{
    "label": "Positive" | "Negative" | "Neutral",
    "confidence": float,
    "explanation": "A detailed explanation justifying the sentiment based on the strict guidelines.",
    "evidence_phrases": ["phrase 1", "phrase 2"]
}}

---
STRICT MODE GUIDELINES:
- Positive: ONLY for reviews with CLEAR, STRONG praise and minimal criticism (e.g., "excellent", "amazing", "loved it").
- Negative: ONLY for reviews with CLEAR, STRONG criticism and minimal praise (e.g., "terrible", "awful", "hated it").
- Neutral: This is the default. Use for mixed reviews, mild language, factual descriptions, or any ambiguity. When in doubt, CHOOSE NEUTRAL.

---
EXAMPLE:
Review: "The lead actor did a decent job and some of the visuals were nice, but the plot was predictable and the ending felt rushed. It was an okay movie."
Your JSON Output:
{{
    "label": "Neutral",
    "confidence": 0.85,
    "explanation": "The review contains both positive comments ('decent job', 'visuals were nice') and negative criticisms ('plot was predictable', 'ending felt rushed'). According to strict guidelines, this mixed sentiment defaults to Neutral.",
    "evidence_phrases": ["decent job", "visuals were nice", "plot was predictable", "ending felt rushed"]
}}
---

Analyze the following movie review according to these strict rules.

Movie Review:
{review_text.strip()}
"""
    else:
        prompt_text = f"""
You are a perceptive film critic assistant. Your task is to analyze a movie review for its sentiment, detecting both explicit and SUBTLE emotional cues.

Your response MUST be a valid JSON object in this exact format:
{{
    "label": "Positive" | "Negative" | "Neutral",
    "confidence": float,
    "explanation": "A detailed explanation justifying the sentiment based on the lenient guidelines.",
    "evidence_phrases": ["phrase 1", "phrase 2"]
}}

---
LENIENT MODE GUIDELINES:
- Positive: Look for ANY positive indicators, including subtle praise, implied satisfaction, or an overall positive tone.
- Negative: Look for ANY negative indicators, including subtle criticism, disappointment, or an overall negative tone.
- Neutral: Only for reviews that are purely factual or perfectly balanced.

---
EXAMPLE:
Review: "I wasn't sure what to expect, but I found myself surprisingly invested in the main character's journey. The film definitely makes you think."
Your JSON Output:
{{
    "label": "Positive",
    "confidence": 0.75,
    "explanation": "The reviewer expresses being 'surprisingly invested' which indicates a positive emotional engagement that exceeded expectations. The phrase 'makes you think' is also generally used to denote a positive, thought-provoking experience. The overall tone is one of pleasant surprise.",
    "evidence_phrases": ["surprisingly invested", "character's journey", "makes you think"]
}}
---

Analyze the following movie review according to these lenient rules.

Movie Review:
{review_text.strip()}
"""
    
    # Set up the AI model with low temperature for consistent results
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={
            "temperature": 0.1,  # Low temperature for consistent, less random responses
            "response_mime_type": "application/json",  # Force JSON output
        },
    )
    
    # Try up to 3 times in case of API hiccups
    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt_text)
            response_text = response.text or ""
            
            parsed_result = json.loads(response_text)
            
            return validate_and_clean_result(parsed_result)
            
        except json.JSONDecodeError as e:
            last_error = f"JSON parsing error: {e}"
            time.sleep(0.5 * (attempt + 1))  # Wait longer between retries
            
        except Exception as e:
            last_error = f"Analysis error: {e}"
            time.sleep(0.8 * (attempt + 1))
    
    # If all retries failed, return safe default
    return {
        "label": "Neutral",
        "confidence": 0.5,
        "explanation": f"Analysis failed after 3 attempts: {last_error}",
        "evidence_phrases": [],
    }

def process_batch_reviews(reviews_list, analysis_mode="lenient", progress_callback=None):
    """Handle multiple reviews at once - useful for batch processing."""
    results = []
    total_reviews = len(reviews_list)
    
    for i, review in enumerate(reviews_list):
        if progress_callback:
            progress_callback(i + 1, total_reviews)
        
        result = analyze_sentiment(review, analysis_mode=analysis_mode)
        results.append(result)
        
        # Small delay between requests to be nice to the API
        if i < total_reviews - 1:
            time.sleep(0.2)
    
    return results

def test_connection():
    """Quick test to make sure everything is working before processing big batches."""
    try:
        test_review = "This movie was absolutely fantastic with amazing acting!"
        result = analyze_sentiment(test_review)
        
        # Check if we got back a valid response structure
        if (result and 
            result.get('label') in ['Positive', 'Negative', 'Neutral'] and
            isinstance(result.get('confidence'), (int, float))):
            return True
        else:
            return False
            
    except Exception:
        return False

# Quick test when running this file directly
if __name__ == "__main__":
    sample_review = "The cinematography was breathtaking and the story kept me engaged throughout!"
    
    print("Testing sentiment analysis...")
    result = analyze_sentiment(sample_review)
    
    print(f"Review: {sample_review}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Make sure everything is working properly
    if test_connection():
        print("✅ System is functioning correctly!")
    else:
        print("❌ Configuration issue detected. Please check your setup.")