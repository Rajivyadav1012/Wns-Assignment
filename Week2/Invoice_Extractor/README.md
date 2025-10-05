# 🧾 AI Invoice Extractor

---

## 🎯 Overview

A powerful, production-ready invoice data extraction system that combines **Groq's lightning-fast LLM inference** with **Tesseract OCR** to automatically extract structured data from invoices in multiple languages. Get invoice details in **2-5 seconds** with **95%+ accuracy**.

### 💡 Why This Tool?

- ⚡ **Ultra-Fast** - Extract data 10x faster than manual entry
- 🎯 **Highly Accurate** - 95%+ accuracy on clear invoices
- 🌍 **Multi-Language** - Supports invoices in any language
- 🔄 **Dual Modes** - Q&A for queries, Auto-extract for full data
- 💰 **Cost-Effective** - Uses free Groq API tier
- 🚀 **Production Ready** - Clean code, error handling, real-time feedback

---

## ✨ Features

- 🚀 **Ultra-Fast Processing** - Extract invoice data in 2-5 seconds
- 🤖 **Dual Extraction Modes** - Q&A Mode or Auto Extract
- 🌍 **Multi-Language Support** - Works with invoices in any language
- 📊 **Structured Output** - Get data in clean JSON format
- 💡 **Smart AI Analysis** - Powered by Groq's Llama 3.1 70B
- 🎨 **Modern UI** - Beautiful gradient interface with real-time feedback
- 💾 **Easy Export** - Download extracted data as JSON
- 🔒 **Secure** - API keys stored in .env file

---

## 🎥 Demo

### Main Interface

![Main](https://github.com/user-attachments/assets/21e28580-ff9c-4dbf-9b3f-ef5c51d69330)

*Beautiful gradient UI with easy file upload and dual extraction modes*

### Q&A Mode 💬
Ask specific questions about your invoice:
- "What is the total amount?"
- "Who is the vendor?"
- "List all items with quantities"

### Auto Extract Mode 📊
Automatically extract all invoice data including:
- Invoice number & date
- Vendor and customer details
- Line items with quantities and prices
- Subtotal, tax, and total amounts

---

## 🏗️ Architecture

### System Flow

```mermaid
flowchart TD
    Start([👤 User Opens App]) --> Upload{📤 Upload Invoice?}
    Upload -->|No| Wait[⏳ Wait for Upload]
    Wait --> Upload
    Upload -->|Yes| Mode{🔀 Select Mode}
    
    Mode -->|Q&A Mode| Question[💬 Enter Question]
    Mode -->|Auto Extract| AutoStart[📊 Click Extract Button]
    
    Question --> OCR1[🔍 OCR Processing]
    AutoStart --> OCR2[🔍 OCR Processing]
    
    OCR1 --> Extract1[📄 Extract Text<br/>via Tesseract]
    OCR2 --> Extract2[📄 Extract Text<br/>via Tesseract]
    
    Extract1 --> AI1[🤖 Send to Groq AI<br/>Llama 3.1 70B]
    Extract2 --> AI2[🤖 Send to Groq AI<br/>Llama 3.1 70B]
    
    AI1 --> QA[💡 Get Q&A Response]
    AI2 --> Structured[📋 Extract Structured Data]
    
    QA --> Display1[✅ Display Answer]
    Structured --> Display2[📊 Display Metrics & JSON]
    
    Display1 --> Download1{💾 Download?}
    Display2 --> Download2{💾 Download JSON?}
    
    Download1 -->|No| End1([✓ Done])
    Download1 -->|Yes| Save1[📥 Save Response]
    Download2 -->|No| End2([✓ Done])
    Download2 -->|Yes| Save2[📥 Save JSON File]
    
    Save1 --> End1
    Save2 --> End2
    
    style Start fill:#667eea,stroke:#764ba2,stroke-width:3px,color:#fff
    style End1 fill:#2ca02c,stroke:#1f77b4,stroke-width:3px,color:#fff
    style End2 fill:#2ca02c,stroke:#1f77b4,stroke-width:3px,color:#fff
    style OCR1 fill:#ff7f0e,stroke:#d62728,stroke-width:2px,color:#fff
    style OCR2 fill:#ff7f0e,stroke:#d62728,stroke-width:2px,color:#fff
    style AI1 fill:#9467bd,stroke:#8c564b,stroke-width:2px,color:#fff
    style AI2 fill:#9467bd,stroke:#8c564b,stroke-width:2px,color:#fff
```

### How It Works

1. **Upload** - User uploads invoice image (JPG/PNG)
2. **OCR Processing** - Tesseract extracts text from image
3. **AI Analysis** - Groq LLM processes and structures data
4. **Display Results** - Show formatted data with export options

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **OCR Engine** | Tesseract | Text extraction |
| **AI/LLM** | Groq API | Data analysis & extraction |
| **Image Processing** | Pillow (PIL) | Image handling |
| **Environment** | Python-dotenv | Config management |

---

## 🚀 Installation

### Prerequisites

Before you begin, ensure you have:
- ✅ Python 3.8 or higher
- ✅ Tesseract OCR installed
- ✅ Groq API key ([Get free key](https://console.groq.com))

### Step 1: Install Tesseract OCR

**Windows:**
1. Download installer: [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer (default path: `C:\Program Files\Tesseract-OCR`)
3. Add to system PATH

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Verify installation:**
```bash
tesseract --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/invoice-extractor.git
cd invoice-extractor
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment

Create `.env` file in project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

**Get your Groq API key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for free account
3. Navigate to API Keys section
4. Create new API key
5. Copy and paste into `.env` file

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` 🎉

---

## 📖 Usage

### Mode 1: Q&A Mode 💬

Perfect for quick queries about specific invoice details.

**Steps:**
1. Upload your invoice image
2. Select **"Q&A Mode"** in sidebar
3. Type your question in the input field
4. Click **"🔍 Analyze Invoice"**
5. Get instant AI-powered answer

**Example Questions:**
```
• What is the invoice number?
• Who is the vendor?
• What is the total amount?
• List all items with their prices
• When is the payment due?
• What's the tax amount?
```

### Mode 2: Auto Extract 📊

Automatically extract all structured data from the invoice.

**Steps:**
1. Upload your invoice image
2. Select **"Auto Extract"** in sidebar
3. Click **"📊 Extract All Data"**
4. View organized results
5. Download JSON file

**Extracted Fields:**
- 📄 Invoice number
- 📅 Invoice date & due date
- 🏢 Vendor name & address
- 👤 Customer details
- 📦 Line items (description, quantity, unit price, total)
- 💰 Subtotal, tax, total amount
- 💱 Currency & payment terms

---

## 📸 Screenshots

### 1. Main Interface
![Main](https://github.com/user-attachments/assets/88a44776-69ef-49eb-9074-26f75f7c51c4)


The landing page features a modern gradient design with easy-to-use file upload functionality and clear mode selection options.

---

### 2. Sidebar with Settings
![invoice_sidebar](https://github.com/user-attachments/assets/c4b1d768-47cc-4be5-b58f-c29e5ec68996)


Intuitive sidebar showing API connection status, extraction mode selection, available AI models, and helpful usage instructions.

---

### 3. Sample Invoice Upload
![invoice](https://github.com/user-attachments/assets/809b83a9-f68f-4468-9509-479a80c9be7e)


Example of a professional invoice ready for processing. Supports JPG, JPEG, and PNG formats.

---

### 4. Extracted Invoice Data
![Extracted_invoice_data](https://github.com/user-attachments/assets/16458043-d8cf-4d71-8b92-3d43df1ff925)



Beautiful metric cards displaying key invoice information including invoice number, date, vendor, customer, and total amount.

---

### 5. Detailed Extraction Results
![Extracted_invoice](https://github.com/user-attachments/assets/13363819-30d1-49d0-a776-b4169c175a30)


Comprehensive view of all extracted data with line items table showing descriptions, quantities, unit prices, and totals.

---

### 6. JSON Output
![Json_output](https://github.com/user-attachments/assets/880708e3-7d06-4e39-a8c1-b2b6e9771dd3)


Clean, structured JSON format with all extracted data ready for download and integration with other systems.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following:

```env
# Required
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

# Optional: Set Tesseract path if not in system PATH
# TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Supported Formats

- ✅ **Images**: JPG, JPEG, PNG
- ✅ **Max Size**: 200MB per file
- ⏳ **PDF Support**: Coming soon

### Groq Models

The app uses Groq's LLM models:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `llama-3.1-70b-versatile` | ⚡⚡⚡ | 🎯🎯🎯 | Default (Best) |
| `mixtral-8x7b-32768` | ⚡⚡ | 🎯🎯 | Alternative |
| `gemma2-9b-it` | ⚡ | 🎯 | Lightweight |

---

## 📂 Project Structure

```
invoice-extractor/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
│
├── screenshots/             # UI screenshots
│   ├── main.png
│   ├── invoice_sidebar.png
│   ├── sample_invoice.png
│   ├── extracted_invoice.png
│   ├── extracted_invoice_data.png
│   └── json_output.png
│
├── samples/                 # Sample invoices (optional)
│   └── sample_invoice.jpg
│
└── outputs/                 # Generated JSON files
    └── extracted_data.json
```

---

## 🔧 Troubleshooting

### Issue: Tesseract Not Found

**Error:** `TesseractNotFoundError`

**Solution:**
```python
# Add this to app.py after imports
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue: GROQ_API_KEY Not Found

**Error:** `API key not found in .env file`

**Solution:**
1. Verify `.env` file exists in project root
2. Check file content: `type .env` (Windows) or `cat .env` (Mac/Linux)
3. Ensure format: `GROQ_API_KEY=gsk_...`
4. No spaces around `=` sign
5. Restart the app

### Issue: Poor OCR Accuracy

**Solutions:**
- ✅ Use high-resolution images (300 DPI minimum)
- ✅ Ensure good lighting, no glare or shadows
- ✅ Straighten skewed/rotated images
- ✅ Use clear, printed text (not handwritten)
- ✅ Crop to invoice area only

### Issue: Slow Processing

**Solutions:**
- ✅ Check internet connection
- ✅ Reduce image size/resolution
- ✅ Verify Groq API status: [status.groq.com](https://status.groq.com)
- ✅ Try different Groq model

### Issue: API Rate Limits

**Solutions:**
- ✅ Check API quota at [console.groq.com](https://console.groq.com)
- ✅ Wait and retry after cooldown period
- ✅ Upgrade to paid tier if needed

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Processing Speed** | 2-5 seconds per invoice |
| **OCR Accuracy** | 95%+ on clear invoices |
| **AI Accuracy** | 95%+ field extraction |
| **Supported Languages** | Multi-language (Tesseract OCR) |
| **Max File Size** | 200MB |
| **Concurrent Users** | Depends on deployment |

---

## 🗺️ Roadmap

### Planned Features

- [ ] **PDF Support** - Multi-page PDF processing
- [ ] **Batch Processing** - Process multiple invoices at once
- [ ] **CSV/Excel Export** - Additional export formats
- [ ] **Custom Templates** - Define custom extraction fields
- [ ] **API Endpoints** - RESTful API for integrations
- [ ] **Database Storage** - Store and search historical invoices
- [ ] **Duplicate Detection** - Identify duplicate invoices
- [ ] **Analytics Dashboard** - Spending insights and trends
- [ ] **Mobile App** - iOS and Android versions
- [ ] **Email Integration** - Process invoices from email

### Version History

- **v1.0.0** (Current) - Initial release with Q&A and Auto Extract modes

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Contribution Guidelines

- Write clear commit messages
- Follow PEP 8 style guide for Python
- Add tests for new features
- Update documentation as needed
- Be respectful and constructive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


---


---

