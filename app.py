import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# -------------------------------------------------
# Streamlit page config – MUST be the first command
# -------------------------------------------------
st.set_page_config(page_title="Conversation Emotion & Sentiment Analyzer", layout="centered")



# ------------------------------
# Load model & tokenizer (CPU)
# ------------------------------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

emotion_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=1,
    device=-1  # -1 = CPU; change to 0 for GPU if available
)

# ------------------------------
# Helper data & functions
# ------------------------------
SUGGESTIONS = {
    "joy": "Keep spreading positivity and enjoy the moment!",
    "love": "Share your love and gratitude with someone today.",
    "surprise": "Be open to new experiences and stay curious!",
    "anger": "Take a deep breath and step away to cool down.",
    "sadness": "Talk to someone you trust and practice self‑care.",
    "fear": "Face your fears slowly and be kind to yourself.",
    "neutral": "Maintain balance and mindfulness in your thoughts."
}

POSITIVE_EMOTIONS = {"joy", "love", "surprise"}


def get_sentiment(emotion: str) -> str:
    """Map emotion → sentiment polarity."""
    return "Positive" if emotion in POSITIVE_EMOTIONS else "Negative"


def analyse_text(text: str):
    """Return emotion, confidence score, sentiment, suggestion for a single message."""
    results = emotion_classifier(text)

    # 🛡️ Handle nested lists like [[{...}]] or flat [{...}]
    top_result = results[0][0] if isinstance(results[0], list) else results[0]

    emotion = top_result["label"].lower()
    score = top_result["score"]
    sentiment = get_sentiment(emotion)
    suggestion = SUGGESTIONS.get(emotion, "Stay balanced and reflect.")
    return emotion, score, sentiment, suggestion


# ------------------------------
# Streamlit UI
# ------------------------------

st.title("🎭 Conversation Emotion & Sentiment Analyzer")

st.markdown(
    "Enter your conversation below, **one message per line** (press **Shift+Enter** for new lines).  "
    "We'll identify the emotion for the **first** and **last** messages and give an **overall** emotion summary."
)

conversation_input = st.text_area("🗨️ Paste conversation here", height=250)

if st.button("🔍 Analyze"):
    # Pre‑process input
    messages = [line.strip() for line in conversation_input.split("\n") if line.strip()]

    if not messages:
        st.warning("Please enter at least one message.")
        st.stop()

    # ------------------------------
    # Analyse first & last messages
    # ------------------------------
    first_emotion, first_score, first_sentiment, first_sugg = analyse_text(messages[0])
    last_emotion, last_score, last_sentiment, last_sugg = analyse_text(messages[-1])

    # ------------------------------
    # Overall analysis
    # ------------------------------
    emotion_counts = {}
    pos_count = neg_count = 0

    for msg in messages:
        emotion, _, sentiment, _ = analyse_text(msg)
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        if sentiment == "Positive":
            pos_count += 1
        else:
            neg_count += 1

    dominant_emotion = max(emotion_counts, key=emotion_counts.get).title()
    overall_sentiment = "Positive" if pos_count >= neg_count else "Negative"
    overall_sugg = SUGGESTIONS.get(dominant_emotion.lower(), "Stay balanced and reflect.")

    # ------------------------------
    # Display results
    # ------------------------------
    st.markdown("---")

    st.subheader("📝 First Message")
    st.write(f"**Detected Emotion:** `{first_emotion.title()}`  (Confidence: `{first_score:.2f}`)")
    st.write(f"**Sentiment Type:** `{first_sentiment}`")
    st.write(f"**Suggestion:** {first_sugg}")

    st.markdown("---")

    st.subheader("📝 Last Message")
    st.write(f"**Detected Emotion:** `{last_emotion.title()}`  (Confidence: `{last_score:.2f}`)")
    st.write(f"**Sentiment Type:** `{last_sentiment}`")
    st.write(f"**Suggestion:** {last_sugg}")

    st.markdown("---")

    st.subheader("✅ Overall Summary")
    st.write(f"**Dominant Emotion:** `{dominant_emotion}`")
    st.write(f"**Overall Sentiment:** `{overall_sentiment}`")
    st.write(f"**Suggestion:** {overall_sugg}")

    st.markdown("---")
    st.info("Analysis complete! Feel free to edit your conversation and run again.")
