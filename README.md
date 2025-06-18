# Conversation-Emotion-Sentiment-Analyzer🧠 
A chat-style emotion and sentiment analysis web app built with Streamlit and Hugging Face Transformers. Users can input full conversations (one line per message), and the application will detect emotional tone and sentiment for the first, last, and overall messages — including real-time wellness suggestions.

✨Features

💬 Accepts full conversations (one message per line)

🔍 Detects emotion and sentiment per message

🧾 Shows first and last message emotion & sentiment

📊 Displays overall dominant emotion and sentiment

💡 Provides suggestions based on emotional tone

🤖 Model Used

🔗 j-hartmann/emotion-english-distilroberta-base

 Emotions detected: joy, love, anger, sadness, fear, surprise, neutral

⚙️ Tech Stack

🐍 Python 3.7+

🔁 Streamlit – interactive UI

🤗 Transformers – NLP pipeline

🔥 PyTorch – backend for the emotion model

🚀 How to Run

🧩 1. Install Required Libraries

pip install streamlit transformers torch

▶️ 2. Launch the App

streamlit run app.py

Visit 👉 http://localhost:8501 in your browser.

✍️ Sample Input

Alex: Hey Sam, how are you feeling today?

Sam: I don’t know... I feel really anxious and overwhelmed.

Alex: I’m here for you, always.

Sam: Thanks. I really needed to hear that.

✅ Output

First Message Emotion: Joy (Sentiment: Positive)

Last Message Emotion: Love (Sentiment: Positive)

Overall: Dominant Emotion - Love, Sentiment - Positive

Suggestion: Share your love and gratitude with someone today.









