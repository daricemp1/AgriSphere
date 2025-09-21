import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

# Transformers
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
)

#Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, that's fine
    pass

# Pretrained AgricultureBERT with the best hyperparameters tuned 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = str(PROJECT_ROOT / "artifacts" / "agri_bert")

# Optional override
LOCAL_DIR = os.getenv("AGRI_BERT_DIR", DEFAULT_ARTIFACT_DIR)

STRICT_LOCAL = os.getenv("AGRI_BERT_STRICT_LOCAL", "0").lower() in ("1", "true", "yes")

def _local_model_available(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))

def load_fill_mask_pipeline():
    """Load from ./artifacts/agri_bert (or AGRI_BERT_DIR). Optionally strict-local; otherwise fallback to Hub."""
    device = 0 if torch.cuda.is_available() else -1

    if _local_model_available(LOCAL_DIR):
        try:
            tok = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)
            mdl = AutoModelForMaskedLM.from_pretrained(LOCAL_DIR, local_files_only=True)
            return pipeline("fill-mask", model=mdl, tokenizer=tok, device=device)
        except Exception as e:
            if STRICT_LOCAL:
                raise RuntimeError(f"Failed to load local AgricultureBERT at {LOCAL_DIR}: {e}") from e
            print(f"[AgricultureBERT] Local load failed ({e}). Falling back to Hub…")
    else:
        if STRICT_LOCAL:
            raise FileNotFoundError(
                f"AgricultureBERT not found at {LOCAL_DIR}. "
                "Place your saved model under ./artifacts/agri_bert or set AGRI_BERT_DIR."
            )

    # Fallback: public Hub model 
    return pipeline(
        "fill-mask",
        model="recobo/agriculture-bert-uncased",
        tokenizer="recobo/agriculture-bert-uncased",
        device=device,
    )

txt_fill = load_fill_mask_pipeline()

def _clean_token(tok: str) -> str:
    """Normalize WordPiece tokens (e.g., '##ing' -> 'ing'), strip non-alpha."""
    t = (tok or "").replace("#", "").strip().lower()
    return t if (t.isalpha() and len(t) >= 3) else ""

def refine_keywords_with_agribert(disease: str, crop_stage: str = None) -> str:
    """
    Uses local AgricultureBERT (if available) to propose a verb-like action for the masked span.
    Falls back to the Hub model transparently unless STRICT_LOCAL=1.
    """
    prompt = f"The crop has {disease} during the {crop_stage or 'growing'} stage and farmers should [MASK]"
    out = txt_fill(prompt, top_k=10)
    for pred in out:
        token = _clean_token(pred.get("token_str", ""))
        if not token:
            continue
        if token in {"not", "have", "do", "be", "rain"}:
            continue
        return token
    return "act"



# LLM advisory via Hugging Face Inference Providers (router)

HUGGINGFACE_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"

# Secure token loading with proper error handling
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError(
        "HUGGINGFACE_TOKEN environment variable is required. "
        "Please set it in your environment or .env file."
    )

def _extract_numeric(val):
    """Strip arrows/symbols, return float if possible, else 0.0"""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    txt = re.sub(r"[^\d\.\-]", "", str(val))
    try:
        return float(txt)
    except Exception:
        return 0.0
def generate_guidelines_via_mistral(disease, weather_summary, crop_stage=None, crop=None):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json",
    }

    # Normalize forecast values (e.g., "25.1 ↗" -> 25.1)
    norm_forecast = {
        "precipitation": _extract_numeric(weather_summary.get("Precipitation (mm)")),
        "temp_max": _extract_numeric(weather_summary.get("Max Temperature (°C)")),
        "temp_min": _extract_numeric(weather_summary.get("Min Temperature (°C)")),
        "windspeed": _extract_numeric(weather_summary.get("Wind Speed (m/s)")),
        "soil_moisture": _extract_numeric(weather_summary.get("Soil Moisture (%)")),
    }

    # Keyword from AgricultureBERT
    keyword = refine_keywords_with_agribert(disease, crop_stage)

    prompt = (
    f"""You are an agricultural advisor. Give precise, **crop-specific** guidance.

    Context
    - Crop: {crop or 'crop'}
    - Problem/Disease: {disease}
    - Stage: {crop_stage or 'growth'}
    - Latest forecast:
    • Precipitation: {norm_forecast['precipitation']:.2f} mm
    • Temp Max: {norm_forecast['temp_max']:.2f} °C
    • Temp Min: {norm_forecast['temp_min']:.2f} °C
    • Windspeed: {norm_forecast['windspeed']:.2f} m/s
    • Soil Moisture: {norm_forecast['soil_moisture']:.2f} %
    - Relevant keyword: '{keyword}'

    Instructions
    1) Provide **2 specific, actionable recommendations for field/commercial growers** that are tailored to the stated crop and disease.
    - Include exact **active ingredient** and **rate with units** (e.g., “copper hydroxide 0.3–0.5 kg/ha in 400 L water”).
    - Add **spray interval**, **PHI** (pre-harvest interval) if applicable, and timing rules using the forecast (e.g., avoid spraying if wind >7–8 m/s; don’t spray within 6–12 h of >10 mm rain).
    - If chemicals are **not** appropriate (e.g., viral diseases), specify **non-chemical controls** (vector control, rogueing, sanitation) with concrete steps.
    - Include any **crop-specific cautions** (e.g., sulfur phytotoxicity on tomato >30–32 °C, copper burn risk on cucurbits, variety sensitivity).

    2) Provide **2 alternatives for home/home-garden growers**, using consumer products **commonly found in hardware/garden stores**.
    - Use **consumer-accessible actives** (e.g., copper soap/copper octanoate, sulfur dust/wettable sulfur, potassium bicarbonate, horticultural oil, neem oil/azadirachtin, insecticidal soap, *Bacillus*-based biofungicides, Bt for caterpillars), chosen to fit the **disease/crop**.
    - Give clear **mixing or ready-to-use directions** (e.g., “mix 10 mL per 1 L water” or “apply RTU until runoff”), simple **frequency**, and basic **safety** (gloves, no spray above 30–32 °C or in wind >7 m/s).
    - Prefer **generic actives** over brand names.

    3) Finish with a short **Reasoning** paragraph that explicitly cites the forecast numbers (rain, temperatures, wind, soil moisture) to justify timing and product choice.

    Output format (Markdown):
    **Recommended Actions (Commercial)**
    - Action 1
    - Action 2

    **Home Grower Alternatives (Store-Available)**
    - Action 1
    - Action 2

    **Reasoning**
    One short paragraph tying your advice to the forecast values.

    Keep it concise, specific, and crop-correct. Do not invent unavailable chemistries or brand names."""
    )


    payload = {
        "model": "deepseek/deepseek-v3-0324",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7,
        "stream": False,
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        elif "error" in result:
            return f"API Error: {result['error']}"
        else:
            return "Unexpected response format"
    except requests.exceptions.RequestException as e:
        return try_fallback_model(disease, norm_forecast, crop_stage, keyword, str(e), crop)
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Fallback text if API fails

def try_fallback_model(disease, weather_summary, crop_stage, keyword, original_error, crop=None):
    rain = weather_summary.get("precipitation", 0.0)
    rain_advice = (
        "ensure good drainage and avoid overhead watering"
        if rain > 0.1 else
        "consider irrigation if needed"
    )

    def _fmt(k, v):
        unit = {"precipitation": "mm", "windspeed": "m/s", "soil_moisture": "%", "temp_max": "°C", "temp_min": "°C"}.get(k, "")
        try:
            return f"{k}: {float(v):.2f} {unit}".strip()
        except Exception:
            return f"{k}: {v} {unit}".strip()

    summary = ", ".join(_fmt(k, weather_summary.get(k, 0.0))
                        for k in ["precipitation", "temp_max", "temp_min", "windspeed", "soil_moisture"])

    return f"""**Recommended Actions:**
- Apply appropriate fungicide or treatment for {disease}
- {rain_advice.capitalize()}

**Reasoning:**
The disease '{disease}' requires targeted treatment during the {crop_stage or 'growth'} stage for {crop or 'the crop'}.
Keyword '{keyword}' implies action is necessary. Weather summary: {summary}. These conditions influence optimal treatment timing.

Note: API unavailable ({original_error[:60]}...). Using fallback recommendations."""
