import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer bundle
bundle = joblib.load("naija_sms_detector_bundle.joblib")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

st.set_page_config(page_title="Naija Scam SMS Detector", page_icon="📱")

st.title("🇳🇬 Naija Scam SMS Detector")
st.markdown("Detect whether an SMS message is **scam** or **legit (ham)** using a trained machine learning model.")

# --- Language Selector ---
language = st.selectbox("🌐 Select Language", ["English", "Yoruba", "Hausa", "Igbo"])
if language != "English":
    st.info("🔄 Multilingual support coming soon. Currently, only English is supported.")

# --- SMS Input ---
sms_text = st.text_area("📩 Paste or type your SMS message here:", height=150)

if st.button("🚀 Analyze SMS"):
    if not sms_text.strip():
        st.warning("Please enter a message first.")
    else:
        # Vectorize input
        vector = vectorizer.transform([sms_text])
        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][1]  # probability of spam
        
        label = "🚨 Scam (Spam)" if pred == 1 else "✅ Legit (Ham)"
        color = "red" if pred == 1 else "green"
        confidence = round(proba * 100, 2) if pred == 1 else round((1 - proba) * 100, 2)

        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence}%")

        # --- Explainability: show top suspicious words ---
        st.subheader("🧠 Explanation")
        feature_names = vectorizer.get_feature_names_out()
        tfidf_vector = vector.toarray()[0]
        top_indices = np.argsort(tfidf_vector)[::-1][:5]
        top_words = [feature_names[i] for i in top_indices if tfidf_vector[i] > 0]

        if top_words:
            st.markdown("These words contributed the most to the prediction:")
            st.markdown(f"🔍 **Suspicious Words:** `{', '.join(top_words)}`")
            st.markdown(
                "💡 *These words are commonly found in scam messages, especially in financial frauds, urgent calls to action, or promotions.*"
            )
        else:
            st.markdown("No strongly suspicious words were detected.")

        # --- Feedback Section ---
        st.subheader("🗣️ Was this prediction correct?")
        feedback = st.radio("Let us know:", ["Yes", "No"], horizontal=True)
        if feedback == "No":
            st.success("Thanks! Your feedback will help improve the system.")

# Footer
st.markdown("---")
st.markdown("Built for 🇳🇬 Naija. Powered by Machine Learning. 🚀")
