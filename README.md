# Zeus: A Simple Neural Chatbot 🤖⚡

Welcome to **Zeus**, your lean‑and‑mean, NLTK + Keras–powered neural chatbot. No fluff, no endless configurations—just the fundamentals you need to train and run your own conversational AI in minutes. Ready to ditch those brittle rule‑based scripts? Let’s get you talking with the future of chatbots.

---

## 🚀 Features

- **Neural Network–Driven**: A 3‑layer feed‑forward model (128 → 64 → output) trained on your custom intents.
- **NLTK Preprocessing**: Tokenization, lemmatization, and bag‑of‑words encoding—and we ignore all the junk (`?`, `!`, `.`, `,`).
- **Easy to Extend**: Simply tweak `intents.json` with new tags, patterns, and responses.
- **Zero Magic**: Plain Python code, no hidden frameworks—perfect for learning or prototyping.

---

## 🛠️ Installation

1. **Clone the repo**  
   git clone https://github.com/your‑username/zeus‑chatbot.git
   cd zeus‑chatbot

2.**Create & activate a virtual environment**
  python3 -m venv venv
  source venv/bin/activate   # macOS/Linux
  venv\Scripts\activate      # Windows

3.**Install dependencies**
  pip install nltk numpy tensorflow keras
  python -m nltk.downloader punkt wordnet

---

## 📂 Project Structure

  zeus-chatbot/
  
  ├── intents.json           # Define your tags, patterns & responses here
  
  ├── train_chatbot.py       # Preprocess data & train the model
  
  ├── run_chatbot.py         # Load model & run interactive loop
  
  ├── words.pkl              # Saved vocabulary (auto‑generated)
  
  ├── classes.pkl            # Saved intent classes (auto‑generated)
  
  ├── chatbotmodel.h5        # Your trained Keras model
  
  └── README.md              # (You’re reading it!)
  

---

## 🏋️ Training the Model

1.**Make sure intents.json is filled with your custom intents.**

2.**Run the training script:**
  python train_chatbot.py
  
3.**Boom—you’ll see training progress for 200 epochs. The model is saved as chatbotmodel.h5, and the pickled vocab & classes.**


---

## 💬 Running the Chatbot

1. ** Once trained, just launch: **
   
   python run_chatbot.py
   
  Type your message, hit Enter, and let Zeus handle the rest. Want to tweak response thresholds or add fancy fallback logic? Go for it—this code is yours to own.

---

## 🤝 Contributing

Feel free to fork and send pull requests! Whether it’s better preprocessing, more advanced models, or integrations (Discord, Slack, webhooks)—let’s push conversational AI forward together.

---

## 📜 License

This project is released under the MIT License. Do whatever you want, just don’t hold us responsible if your bot starts philosophizing about the meaning of life. 😉



