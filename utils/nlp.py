import requests
import re
from transformers import pipeline
import os

# 1. AgricultureBERT remains unchanged
txt_fill = pipeline(
    "fill-mask",
    model="recobo/agriculture-bert-uncased",
    tokenizer="recobo/agriculture-bert-uncased"
)

def refine_keywords_with_agribert(disease: str, crop_stage: str = None) -> str:
    prompt = f"The crop has {disease} during the {crop_stage or 'growing'} stage and farmers should [MASK]"
    out = txt_fill(prompt, top_k=10)
    for pred in out:
        token = pred.get("token_str", "").strip().lower()
        if token in {"not", "have", "do", "be", "rain"}:
            continue
        if not token.isalpha() or len(token) < 3:
            continue
        return token
    return "act"

# 2. Updated to use new Hugging Face Inference Providers API
HUGGINGFACE_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

def generate_guidelines_via_mistral(disease, rain, crop_stage=None):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    rain_desc = "rain is expected" if rain else "no rain is expected"
    keyword = refine_keywords_with_agribert(disease, crop_stage)

    # Using  prompt structure 
    prompt = (
        f"Instruction: You are an agricultural expert. A crop is affected by {disease} during the {crop_stage or 'growth'} stage. "
        f"{rain_desc}. Relevant keyword: {keyword}. Provide 1–2 specific, accurate farming actions to address the issue "
        f"(e.g., 'Apply copper-based fungicide at 2 kg/ha', 'Improve drainage'). Avoid vague advice like 'monitor crops'. "
        f"After the actions, provide a brief reasoning section explaining why these actions are suitable, referencing the disease, crop stage, and rain. "
        f"Format the response as follows:\n\n"
        f"**Recommended Actions:**\n- Action 1\n- Action 2\n\n**Reasoning:**\nExplanation here.\n\n"
        f"Example for early blight, flowering stage, rain expected:\n"
        f"**Recommended Actions:**\n- Apply copper-based fungicide at 1.5 kg/ha.\n- Remove lower infected leaves.\n\n"
        f"**Reasoning:**\nCopper-based fungicide controls early blight by inhibiting fungal spore germination, especially critical in wet conditions. "
        f"Removing infected leaves reduces spore spread during the flowering stage, when the crop is vulnerable.\n\n"
        f"Response:"
    )

    payload = {
        "model": "deepseek/deepseek-v3-0324",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)
        
        # Debug information
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")
        
        response.raise_for_status()
        
        result = response.json()
        
        # Handle API errors
        if "error" in result:
            error_msg = result["error"]
            if "currently loading" in error_msg.lower():
                return "⚠️ Model is loading. Please wait 1-2 minutes and try again."
            elif "rate limit" in error_msg.lower():
                return "⚠️ Rate limit exceeded. Please wait and try again."
            return f"⚠️ API Error: {error_msg}"
        
        # Extract the response from chat completion format
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ Unexpected response format"
            
    except requests.exceptions.RequestException as e:
        # Fallback to a simpler model if the main one fails
        return try_fallback_model(disease, rain, crop_stage, keyword, str(e))
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

def try_fallback_model(disease, rain, crop_stage, keyword, original_error):
    """Fallback to a simple local response if API fails"""
    rain_advice = "ensure good drainage and avoid overhead watering" if rain else "consider irrigation if needed"
    
    basic_advice = f"""**Recommended Actions:**
- Apply appropriate fungicide or treatment for {disease}
- {rain_advice.capitalize()}

**Reasoning:**
{disease} requires targeted treatment during the {crop_stage or 'growth'} stage. The recommended keyword '{keyword}' suggests specific action is needed. Weather conditions (rain: {rain}) influence treatment timing and application methods.

⚠️ Note: API unavailable ({original_error[:50]}...). Using basic recommendations."""
    
    return basic_advice