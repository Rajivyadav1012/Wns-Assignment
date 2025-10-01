# 🤖 AI Chat Assistant with Groq

A fast AI chat application built with Streamlit and Groq API. Features multiple AI models, conversation management, and a modern UI.


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


---

## 🎯 Usage

### Basic Chat
1. Type message in input box
2. Press Enter to send
3. AI responds in real-time


<img width="1350" height="635" alt="image" src="https://github.com/user-attachments/assets/9d6ef58b-c91a-4692-bf80-03fe75a4aa76" />


### Model Selection
Choose from 5 models in sidebar:
- **Llama 3.3 70B** - Best quality
- **Llama 3.1 70B** - Fast & smart
- **Llama 3.1 8B** - Ultra fast
- **Mixtral 8x7B** - Long context (32k tokens)
- **Gemma 2 9B** - Balanced

<img width="247" height="383" alt="image" src="https://github.com/user-attachments/assets/591490d1-e43c-4a1e-9f76-00f801590a5c" />


### Save & Load Chats
- **Save Chat** - Store current conversation
- **New Chat** - Start fresh (auto-saves previous)
- **Load Chat** - Access saved conversations
- **Delete** - Remove individual chats
- **Clear All** - Remove all saved chats

<img width="272" height="414" alt="image" src="https://github.com/user-attachments/assets/d1dc0ac6-7dc1-404f-91e5-54a57665c4c0" />


### Settings
- **Temperature** (0.0-2.0) - Creativity level
- **Max Tokens** (256-8000) - Response length
- **Stream Response** - Toggle word-by-word display



### Export
Download conversations as JSON with timestamps.


---

## 📊 Statistics

Track your usage in sidebar:
- Total messages
- Saved chats
- API requests
- Your messages



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

### Conversation Example

<img width="1353" height="630" alt="image" src="https://github.com/user-attachments/assets/55c30b58-8248-4215-a70d-579fc3635b26" />

