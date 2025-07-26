
import streamlit as st
from PIL import Image

# import all the custom functions 
from utils.vision   import load_vision_model, predict_disease
from utils.forecast import fetch_weather, predict_rain
from utils.nlp      import generate_guidelines_via_mistral

# Load fine-tuned EfficientNet + labels
st.set_page_config(page_title=" Multimodal Farming Assistant")
vision_model_bundle = load_vision_model() 

# Streamlit UI
st.title("Welcome to AgriSphere")

# Step 1: Upload crop image
img_file = st.file_uploader("📷 Upload a crop leaf image", type=["png", "jpg", "jpeg"])

# Step 2: Crop stage selection
crop_stage = st.selectbox("🌱 Select Crop Stage", ["", "germination", "vegetative", "flowering", "maturity", "harvest"])

# Step 3: Inference pipeline
if img_file:
    st.image(Image.open(img_file), caption="Input Leaf", use_container_width=True)

    # 1. Vision --> disease prediction
    disease = predict_disease(img_file, vision_model_bundle)
    st.success(f"🔍 Detected disease: **{disease}**")

    # 2. Forecast → rain prediction
    df = fetch_weather()
    rain = predict_rain(df)
    model, forecast, rain_bool, rain_amt = predict_rain(df)

    st.write(f"🌧️  Predicted total rain (next 48h): **{rain_amt:.2f} mm**")
    st.write("⚠️ Heavy rain expected?", "**Yes**" if rain_bool else "**No**")

    #Now you have `model` and `forecast` in scope
    fig = model.plot(forecast)
    st.pyplot(fig)

    # 3. Generate advice
    if st.button("💡 Get Recommendation"):
        st.subheader("💬 Farming Advice")

        final_advice = generate_guidelines_via_mistral(
            disease=disease,
            rain=rain,
            crop_stage=crop_stage if crop_stage else None
        )

        st.success(final_advice)

else:
    st.info("Please upload a crop leaf image to start.")
