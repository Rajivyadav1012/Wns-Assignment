# 🤖 AI Chat Assistant with Groq

A fast AI chat application built with Streamlit and Groq API. Features multiple AI models, conversation management, and a modern UI.

![Main Interface](screenshots/main-interface.png)

## ✨ Features

- **Lightning Fast Responses** - Powered by Groq API
- **5 AI Models** - Llama 3.3, Llama 3.1, Mixtral, Gemma
- **Save & Load Chats** - Manage multiple conversations
- **Export to JSON** - Download your chat history
- **Streaming Responses** - Real-time word-by-word display
- **Customizable Settings** - Temperature, tokens, streaming

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/groq-ai-chat.git
cd groq-ai-chat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup API Key
Create a `.env` file:
```
GROQ_API_KEY=gsk_your_api_key_here
```

### 4. Run Application
```bash
streamlit run app.py
```

---

## 🔑 Get Your Free Groq API Key

1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up (no credit card required)
3. Go to "API Keys" → "Create API Key"
4. Copy key and add to `.env` file

**Free Tier:**
- 30 requests/minute
- 14,400 requests/day

![API Setup](screenshots/api-setup.png)

---

## 🎯 Usage

### Basic Chat
1. Type message in input box
2. Press Enter to send
3. AI responds in real-time

![Chat Interface](screenshots/chat-interface.png)

### Model Selection
Choose from 5 models in sidebar:
- **Llama 3.3 70B** - Best quality
- **Llama 3.1 70B** - Fast & smart
- **Llama 3.1 8B** - Ultra fast
- **Mixtral 8x7B** - Long context (32k tokens)
- **Gemma 2 9B** - Balanced

![Model Selection](screenshots/model-selection.png)

### Save & Load Chats
- **Save Chat** - Store current conversation
- **New Chat** - Start fresh (auto-saves previous)
- **Load Chat** - Access saved conversations
- **Delete** - Remove individual chats
- **Clear All** - Remove all saved chats

![Saved Chats](screenshots/saved-chats.png)

### Settings
- **Temperature** (0.0-2.0) - Creativity level
- **Max Tokens** (256-8000) - Response length
- **Stream Response** - Toggle word-by-word display

![Settings](screenshots/settings.png)

### Export
Download conversations as JSON with timestamps.

![Export](screenshots/export.png)

---

## 📊 Statistics

Track your usage in sidebar:
- Total messages
- Saved chats
- API requests
- Your messages

![Statistics](screenshots/statistics.png)

---

## 🔍 Troubleshooting

### API Key Error
```
⚠️ GROQ_API_KEY not found
```
**Fix:** Create `.env` file with your API key

### Authentication Failed
```
❌ Authentication failed
```
**Fix:** Verify API key is correct, no extra spaces

### Rate Limit
```
⚠️ Rate limit reached
```
**Fix:** Wait a moment (30 req/min limit)

---

## 📦 Requirements

```txt
streamlit>=1.28.0
groq>=0.4.0
python-dotenv>=1.0.0
httpx>=0.24.0
urllib3>=2.0.0
```

---

## 📸 More Screenshots

### Full Interface
![Full Interface](screenshots/full-interface.png)

### Conversation Example
![Conversation](screenshots/conversation.png)

### Mobile View
![Mobile View](screenshots/mobile-view.png)

---

## 📝 License

MIT License - Feel free to use and modify

---

## 🤝 Contributing

Issues and pull requests welcome!

---

**Made with ❤️ using Streamlit and Groq**
