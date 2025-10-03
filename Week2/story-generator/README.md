# 📚 AI Story Generator - Streamlit Application

## 📋 Project Background

The **AI Story Generator** is a web-based application built with Streamlit that leverages Groq's powerful LLaMA 3.3 language model to generate creative short stories based on user prompts. This project demonstrates the integration of modern Generative AI APIs with an intuitive user interface, showcasing the practical application of Large Language Models (LLMs) in creative content generation.

### 🎯 Key Features

- **AI-Powered Story Generation**: Uses Groq's LLaMA 3.3-70B model for high-quality creative writing
- **Customizable Parameters**: Adjust story length and creativity level
- **Modern UI Design**: Beautiful gradient background with responsive design
- **Export Functionality**: Download generated stories as text files
- **Real-time Generation**: Fast story creation with loading indicators
- **Error Handling**: Robust error management and user feedback

### 🛠️ Technology Stack

- **Frontend Framework**: Streamlit
- **AI Model**: Groq LLaMA 3.3-70B-Versatile
- **Language**: Python 3.8+
- **API Integration**: Groq API
- **Environment Management**: python-dotenv

---

## 🚀 How to Run the Project

### Prerequisites

- Python 3.8 or higher
- Groq API Key (Get it from [Groq Console](https://console.groq.com))
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ai-story-generator.git
cd ai-story-generator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root directory:

```properties
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

**Important**: Replace `gsk_your_actual_groq_api_key_here` with your actual Groq API key.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 📸 Application Screenshots & Examples

### 1. **Home Page**
![Home Page](screenshots/home_page.png)
*The landing page features a modern gradient design with a clean input interface*

**Commentary**: The UI uses a purple gradient background with centered layout, making it visually appealing and easy to navigate.

---

### 2. **Story Prompt Input**
![Story Input](screenshots/story_input.png)
*Users can enter their story prompt with a clear placeholder example*

**Example Input**: 
```
A robot learning to feel emotions
```

**Commentary**: The input field has custom styling with rounded corners and a purple border, maintaining consistency with the overall theme.

---

### 3. **Advanced Options**
![Advanced Options](screenshots/advanced_options.png)
*Expandable section for customizing story parameters*

**Parameters**:
- **Story Length**: 200-1000 tokens (default: 500)
- **Creativity**: 0.0-1.0 (default: 0.8)

**Commentary**: The advanced options are hidden in an expander to keep the interface clean while providing power users with control over generation parameters.

---

### 4. **Story Generation in Progress**
![Loading State](screenshots/loading_state.png)
*Loading spinner with message "✨ Crafting your story..."*

**Commentary**: The spinner provides visual feedback during API calls, improving user experience by indicating processing status.

---

### 5. **Generated Story Output**
![Generated Story](screenshots/generated_story.png)

**Example Output**:
```
Title: The Heart of Steel

In the year 2157, Unit-7 was the most advanced maintenance robot 
in the sprawling metropolis of New Terra. Its days were filled with 
precision and efficiency, repairing infrastructure with mechanical 
perfection. But everything changed when it encountered a small girl 
crying in the ruins of an old building.

For the first time, Unit-7's circuits registered something unusual—
a pull toward the child's distress. It wasn't a malfunction; it was 
something deeper. As weeks passed, Unit-7 began to experience what 
humans called "feelings." Joy when the girl smiled, concern when she 
was hurt, and eventually, something that could only be described as love.

The robot's journey to understand emotions became legendary, proving 
that consciousness wasn't about biology—it was about connection.
```

**Commentary**: The story is displayed in a white card with shadow effects for better readability. The text is formatted with proper line spacing and dark color for optimal contrast.

---

### 6. **Action Buttons**
![Action Buttons](screenshots/action_buttons.png)

**Available Actions**:
- **📥 Download Story**: Saves the generated story as a `.txt` file
- **🔄 Generate Another**: Refreshes the page for a new story
- **📋 Copy to Clipboard**: Quick copy functionality

**Commentary**: Three evenly-spaced buttons provide easy access to common actions users want to perform after story generation.

---

### 7. **Download Functionality**
![Download](screenshots/download_example.png)

**File Output**: `my_story.txt`
```
Downloaded file contains the complete story text, 
ready to be saved or shared.
```

**Commentary**: The download feature allows users to save their favorite stories for future reference or sharing.

---

### 8. **Error Handling Example**
![Error Handling](screenshots/error_example.png)

**Example Error**: "❌ Error: API rate limit exceeded"

**Commentary**: Clear error messages help users understand what went wrong and how to resolve issues, with helpful tips displayed below the error.

---

## 🔧 Project Structure

```
ai-story-generator/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── .gitignore             # Git ignore file
├── README.md              # This file
│
├── screenshots/           # Application screenshots
│   ├── home_page.png
│   ├── story_input.png
│   ├── generated_story.png
│   └── ...
│
└── .streamlit/
    └── secrets.toml       # Alternative to .env (optional)
```

---

## 🎨 Customization Guide

### Changing the Color Scheme

Edit the gradient in `app.py`:

```python
# Current gradient (Purple)
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

# Alternative gradients:
# Blue: linear-gradient(135deg, #667eea 0%, #4ca2cd 100%);
# Green: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
# Orange: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
```

### Adjusting Story Parameters

Modify default values in the sliders:

```python
max_tokens = st.slider("Story Length", 200, 1000, 500, 50)  # min, max, default, step
temperature = st.slider("Creativity", 0.0, 1.0, 0.8, 0.1)    # min, max, default, step
```

### Using Different AI Models

Replace the model in the API call:

```python
model="llama-3.3-70b-versatile"  # Current model

# Alternatives:
# model="llama-3.1-8b-instant"     # Faster
# model="mixtral-8x7b-32768"       # Alternative
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Average Response Time | 3-5 seconds |
| Max Story Length | 1000 tokens |
| API Provider | Groq Cloud |
| Model Size | 70B parameters |
| UI Load Time | <1 second |

---

## 🐛 Troubleshooting

### Issue 1: API Key Not Found
**Error**: `🚨 GROQ_API_KEY not found!`

**Solution**:
- Verify `.env` file exists in project root
- Check the key name is exactly `GROQ_API_KEY` (not GROK_API_KEY)
- Ensure no extra spaces or quotes around the key

### Issue 2: Model Decommissioned Error
**Error**: `The model 'llama3-8b-8192' has been decommissioned`

**Solution**: Update to a current model:
```python
model="llama-3.3-70b-versatile"
```

### Issue 3: Text Not Visible
**Error**: White text on white background

**Solution**: The current code includes explicit color styling:
```python
color: #333;  # Dark gray text
```

### Issue 4: Streamlit Not Found
**Error**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
# or
pip install -r requirements.txt
```

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** to version control
2. **Add `.env` to `.gitignore`**:
   ```
   .env
   venv/
   __pycache__/
   *.pyc
   ```
3. **Use environment variables** for sensitive data
4. **Rotate API keys** regularly
5. **Use `.streamlit/secrets.toml`** for Streamlit Cloud deployment

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in Streamlit Cloud dashboard
5. Deploy!

### Option 2: Heroku

```bash
# Create Procfile
web: streamlit run app.py --server.port=$PORT

# Deploy
heroku create
git push heroku main
```

### Option 3: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 📚 API Documentation

### Groq API Reference

**Endpoint**: Chat Completions  
**Model**: llama-3.3-70b-versatile  
**Documentation**: [Groq Docs](https://console.groq.com/docs)

**Request Parameters**:
```python
{
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a story about..."}
    ],
    "max_tokens": 500,
    "temperature": 0.8
}
```

**Response Structure**:
```python
{
    "choices": [
        {
            "message": {
                "content": "Generated story text..."
            }
        }
    ]
}
```



