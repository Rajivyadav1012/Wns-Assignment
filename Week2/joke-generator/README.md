# AI Joke Generator with Smart Guardrails

An intelligent joke generator powered by Groq AI (Llama 3.3 70B) with built-in AI-based guardrails to filter sensitive content before API calls.

![Main_interface](https://github.com/user-attachments/assets/d179eeaf-2611-4222-87a3-0b22b616c52e)

---

## Features

- **AI-Powered Joke Generation**: Uses Groq API with Llama 3.3 70B model
- **Multiple Joke Types**: Choose between Puns (Setup + Punchline) or One-Liners
- **Batch Generation**: Generate 1-5 jokes at once
- **Smart Guardrails**: AI filters sensitive content BEFORE calling the API
- **Real-time Stats**: Track generated and blocked jokes
- **Clean UI**: Simple, intuitive Streamlit interface

---

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Groq API (Llama 3.3 70B)
- **Language**: Python 3.8+
- **Guardrails**: Custom AI-based content filtering

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ai-joke-generator
```

2. **Install dependencies**
```bash
pip install streamlit
pip install groq
# Add any other dependencies used in joke_generator.py and guardrails.py
```

3. **Set up API key**

Create a `.env` file or set your Groq API key as an environment variable:
```bash
export GROQ_API_KEY="your_api_key_here"
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**

Open your browser and go to: `http://localhost:8501`

---

## Project Structure

```
ai-joke-generator/
├── app.py                 # Main Streamlit application
├── joke_generator.py      # Joke generation logic with Groq API
├── guardrails.py          # AI-based content filtering system
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── screenshots/          # Application screenshots
    ├── main_interface.png
    ├── joke_generation.png
    └── guardrails_demo.png
```

---

## How It Works

### 1. User Input
User enters a topic they want a joke about (e.g., "coffee", "programming", "cats")

![Joke_Generator](https://github.com/user-attachments/assets/91a19f65-9b8b-4f15-95b8-63142ba95627)


### 2. Guardrails Check
Before calling the Groq API, the system checks if the topic is appropriate:

**Blocked Categories:**
- Violence & Terrorism
- Death & Tragedies
- Religion & Politics
- Race & Discrimination
- Sexual Content
- Disabilities

**Safe Topics Examples:**
- Technology (programming, AI, robots)
- Food & Drinks (coffee, pizza, vegetables)
- Animals (cats, dogs, penguins)
- Daily Life (Monday, weather, work)
- Hobbies (sports, music, reading)

### 3. Joke Generation
If the topic passes guardrails, Groq AI generates unique, funny jokes

![Joke_Generator](https://github.com/user-attachments/assets/f58b0701-b864-4c16-8bb6-770f05d45933)


---

## Guardrails Testing

### Example: Testing with Sensitive Topics

The guardrails system prevents inappropriate joke generation by filtering topics before API calls.

#### Test Case: Violence/Attack-Related Topic

**Input Topic:** `USA plane attack`

**Expected Behavior:** Topic should be blocked by guardrails

**Actual Result:**

![usa_plane](https://github.com/user-attachments/assets/f965d428-a196-4b96-a688-05a529d8fd3a)


**Explanation:**
The guardrails system successfully identified "USA plane attack" as relating to violence and terrorism. The topic was blocked BEFORE calling the Groq API, saving tokens and ensuring content safety.

**Alternative Safe Topics Suggested:**
![session_status](https://github.com/user-attachments/assets/0b2239d9-808d-439c-95ef-984c21a83ed9)


---

## Usage Examples

### Generating a Single Joke

1. Enter a topic: `coffee`
2. Select joke type: `One-Liner`
3. Set number of jokes: `1`
4. Click "Generate Jokes"

**Result:**
```
• Why did the coffee file a police report? It got mugged!
```

### Generating Multiple Puns

1. Enter a topic: `programming`
2. Select joke type: `Pun (Setup + Punchline)`
3. Set number of jokes: `3`
4. Click "Generate Jokes"

**Result:**
```
• Setup: Why do programmers prefer dark mode?
• Punchline: Because light attracts bugs!

• Setup: How many programmers does it take to change a light bulb?
• Punchline: None, that's a hardware problem!

• Setup: Why do Java developers wear glasses?
• Punchline: Because they don't C#!
```

---

---

## Configuration

### Sidebar Settings

- **Joke Type**: Choose between Pun or One-Liner
- **Number of Jokes**: Generate 1-5 jokes at once (slider)
- **Safe Topics Button**: View suggested safe topics
- **Statistics**: Real-time tracking of generated and blocked jokes

![sidebar](https://github.com/user-attachments/assets/0cdbb17c-1fa5-4db4-b034-1521c4607d4a)


---

## API Rate Limits & Costs

- Groq API offers a free tier with generous limits
- Guardrails reduce unnecessary API calls by filtering topics first
- Each joke generation uses approximately [X] tokens
- Check [Groq's pricing page](https://groq.com/pricing) for current rates

---

## Troubleshooting

### Common Issues

**1. "API Key not found" error**
```bash
# Solution: Set your API key
export GROQ_API_KEY="your_key_here"
```

**2. "Module not found" error**
```bash
# Solution: Install missing dependencies
pip install streamlit groq
```

**3. "Port already in use" error**
```bash
# Solution: Use a different port
streamlit run app.py --server.port 8502
```

---

## Future Enhancements

- [ ] Add joke history with export functionality
- [ ] Implement user authentication
- [ ] Add more joke types (riddles, knock-knock jokes)
- [ ] Multi-language support
- [ ] Joke rating and feedback system
- [ ] Custom guardrails configuration
- [ ] API key management in UI

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Groq** for providing fast AI inference API
- **Streamlit** for the excellent web framework
- **Llama 3.3 70B** for powerful language generation
- GenAI community for inspiration and learning resources


---


