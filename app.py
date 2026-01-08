import streamlit as st
import requests
import base64
import json
import pandas as pd
import altair as alt
from PIL import Image
from io import BytesIO

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="NutriSnap AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets safely
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please set up your .streamlit/secrets.toml")
    st.stop()

MODEL_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-preview-09-2025:generateContent"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #FF4B4B; 
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1 {
        color: #1f1f1f; 
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "daily_goal" not in st.session_state:
    st.session_state.daily_goal = {
        "calories": 2000,
        "proteinG": 120,
        "carbsG": 250,
        "fatG": 65
    }

# ---------------- HELPERS ----------------
def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def create_donut_chart(protein, carbs, fat):
    source = pd.DataFrame({
        "Category": ["Protein", "Carbs", "Fat"],
        "Value": [protein, carbs, fat]
    })
    
    base = alt.Chart(source).encode(
        theta=alt.Theta("Value", stack=True)
    )
    
    pie = base.mark_arc(outerRadius=100, innerRadius=60).encode(
        color=alt.Color("Category", scale=alt.Scale(domain=["Protein", "Carbs", "Fat"], range=["#36a2eb", "#ffcd56", "#ff6384"])),
        order=alt.Order("Value", sort="descending"),
        tooltip=["Category", "Value"]
    )
    
    text = base.mark_text(radius=120).encode(
        text="Value",
        order=alt.Order("Value", sort="descending"),
        color=alt.value("black")
    )
    
    return pie + text

def call_gemini(image_b64: str):
    system_prompt = """
    You are NutriSnap AI, an expert food analyst.
    Analyze the image of the meal and return a JSON object STRICTLY following this schema:
    {
      "foodName": "string",
      "cuisineType": "string",
      "calories": number,
      "macronutrients": { "proteinG": number, "carbsG": number, "fatG": number },
      "insightSummary": "string (short, engaging 1-sentence summary)",
      "dietaryTags": ["string (e.g. Vegan, Keto, Gluten-Free, High-Protein)"],
      "recipe": {
        "title": "string",
        "ingredients": ["string (with quantities)"],
        "instructions": ["string"]
      },
      "allergenAlert": {
        "riskLevel": "Low | Medium | High",
        "detected": ["string"],
        "advice": "string"
      }
    }
    """
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Analyze this food photo and return structured nutrition data."},
                {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }

    response = requests.post(
        f"{MODEL_URL}?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])

# ---------------- SIDEBAR GOALS ----------------
with st.sidebar:
    st.header("⚙️ Profile & Goals")
    st.markdown("Set your daily nutritional targets to see how this meal fits in.")
    
    with st.expander("🎯 Edit Daily Goals", expanded=True):
        st.session_state.daily_goal["calories"] = st.number_input("Calories (kcal)", value=st.session_state.daily_goal["calories"])
        st.session_state.daily_goal["proteinG"] = st.number_input("Protein (g)", value=st.session_state.daily_goal["proteinG"])
        st.session_state.daily_goal["carbsG"] = st.number_input("Carbs (g)", value=st.session_state.daily_goal["carbsG"])
        st.session_state.daily_goal["fatG"] = st.number_input("Fat (g)", value=st.session_state.daily_goal["fatG"])
    
    st.info("💡 **Tip:** Clear photos with good lighting give the best accuracy.")

# ---------------- MAIN UI ----------------
st.title("🍽️ NutriSnap AI")
st.markdown("### Instant Meal Decoding & Nutrition Analysis")

col1, col2 = st.columns([1, 1.5], gap="large")

# ---------------- LEFT PANEL (INPUT) ----------------
with col1:
    st.markdown("#### 📸 Snap Your Meal")
    
    input_method = st.radio("Input method", ["Camera", "Upload"], horizontal=True, label_visibility="collapsed")
    
    image = None
    if input_method == "Camera":
        camera_image = st.camera_input("Take a photo")
        if camera_image: image = Image.open(camera_image)
    else:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file: image = Image.open(uploaded_file)

    if image:
        st.image(image, caption="Ready for analysis", use_container_width=True, channels="RGB")
        
        # Analyze Button
        if st.button("⚡ Decode Nutrition"):
            with st.status("🔍 Analyzing food matrix...", expanded=True) as status:
                try:
                    st.write("Encoding image...")
                    img_b64 = image_to_base64(image)
                    st.write("Consulting AI Chef...")
                    st.session_state.analysis = call_gemini(img_b64)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Analysis Failed", state="error")
                    st.error(f"Error: {e}")

# ---------------- RIGHT PANEL (OUTPUT) ----------------
with col2:
    data = st.session_state.analysis
    goals = st.session_state.daily_goal

    if not data:
        st.markdown(
            """
            <div style="text-align: center; padding: 50px; background: #f0f2f6; border-radius: 10px; color: #666;">
                <h3>👈 Snap a photo to begin</h3>
                <p>AI will identify ingredients, calculate macros, and generate a recipe.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        # Header Section
        st.markdown(f"## {data['foodName']}")
        
        # Badges
        tags = [data['cuisineType']] + data.get('dietaryTags', [])
        st.markdown(" ".join([f"`{tag}`" for tag in tags]))
        
        st.caption(f"📝 {data['insightSummary']}")
        
        st.divider()

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Nutrition", "👨‍🍳 Recipe", "⚠️ Health & Safety"])

        # --- TAB 1: NUTRITION ---
        with tab1:
            # Calories Hero
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Energy", f"{data['calories']} kcal", f"{round((data['calories']/goals['calories'])*100)}% of Daily Goal")
            
            with c2:
                # Donut Chart
                macros = data["macronutrients"]
                chart = create_donut_chart(macros['proteinG'], macros['carbsG'], macros['fatG'])
                st.altair_chart(chart, use_container_width=True)

            # Detailed Macro Cards
            st.markdown("#### Macro Breakdown")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                with st.container(border=True):
                    st.metric("Protein", f"{macros['proteinG']}g", border=True)
                    st.progress(min(macros["proteinG"] / goals["proteinG"], 1.0))
            with mc2:
                with st.container(border=True):
                    st.metric("Carbs", f"{macros['carbsG']}g", border=True)
                    st.progress(min(macros["carbsG"] / goals["carbsG"], 1.0))
            with mc3:
                with st.container(border=True):
                    st.metric("Fat", f"{macros['fatG']}g", border=True)
                    st.progress(min(macros["fatG"] / goals["fatG"], 1.0))

        # --- TAB 2: RECIPE ---
        with tab2:
            recipe = data["recipe"]
            st.subheader(recipe['title'])
            
            col_ing, col_inst = st.columns(2)
            
            with col_ing:
                st.markdown("**🛒 Ingredients**")
                for ing in recipe["ingredients"]:
                    st.markdown(f"- {ing}")
            
            with col_inst:
                st.markdown("**🍳 Instructions**")
                for idx, step in enumerate(recipe["instructions"], 1):
                    st.markdown(f"**{idx}.** {step}")

        # --- TAB 3: ALERTS ---
        with tab3:
            alert = data["allergenAlert"]
            
            risk_color = {
                "High": "red",
                "Medium": "orange",
                "Low": "green"
            }
            
            st.markdown(f"**Risk Level:** :{risk_color[alert['riskLevel']]}[{alert['riskLevel']}]")
            
            if alert['riskLevel'] != "Low":
                st.warning(alert['advice'])
            else:
                st.success(alert['advice'])
                
            if alert["detected"]:
                st.write("Potential Allergens Detected:")
                for item in alert["detected"]:
                    st.markdown(f"- 🔴 **{item}**")