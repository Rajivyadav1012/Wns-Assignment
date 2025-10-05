"""
Multi-Language Invoice Extractor using Groq API
Streamlit App with OCR + Groq LLM (Using .env file)
"""

import streamlit as st
from PIL import Image
import pytesseract
from groq import Groq
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("api_key")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file!")
    st.info("Please add GROQ_API_KEY=your_key_here to your .env file")
    st.stop()

# Configure Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from image using OCR
def extract_text_from_image(uploaded_file):
    """Extract text from uploaded image using Tesseract OCR"""
    try:
        image = Image.open(uploaded_file)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

# Function to get Groq response
def get_groq_response(input_prompt, extracted_text, user_question):
    """Get response from Groq API"""
    try:
        # Combine the prompts
        full_prompt = f"""
{input_prompt}

Extracted Invoice Text:
{extracted_text}

User Question: {user_question}

Please provide a detailed and accurate response based on the invoice text above.
"""
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert invoice analyst who can extract and interpret information from invoice text accurately."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            model="llama-3.1-70b-versatile",  # or "mixtral-8x7b-32768"
            temperature=0.1,
            max_tokens=2000
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error getting Groq response: {str(e)}")
        return None

# Function to extract structured invoice data
def extract_invoice_data(extracted_text):
    """Extract structured data from invoice using Groq"""
    try:
        prompt = f"""
Extract the following information from this invoice text and return it in JSON format:

Invoice Text:
{extracted_text}

Please extract:
- invoice_number: The invoice number
- invoice_date: Date of invoice
- vendor_name: Vendor/supplier name
- customer_name: Customer/buyer name
- line_items: List of items with description, quantity, price
- subtotal: Subtotal amount
- tax: Tax amount
- total_amount: Total amount
- currency: Currency used

Return ONLY valid JSON, no additional text.
"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction expert. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=2000
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            invoice_data = json.loads(response_text)
            return invoice_data
        except json.JSONDecodeError:
            st.warning("Could not parse structured data, showing raw response")
            return {"raw_response": response_text}
    
    except Exception as e:
        st.error(f"Error extracting structured data: {str(e)}")
        return None

# Initialize Streamlit app
st.set_page_config(
    page_title="Invoice Extractor - Groq",
    page_icon="🧾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧾 Multi-Language Invoice Extractor</div>', unsafe_allow_html=True)
st.markdown("*Powered by Groq AI & Tesseract OCR*")

# Show API key status
if GROQ_API_KEY:
    st.sidebar.success("✅ API Key Loaded from .env")
else:
    st.sidebar.error("❌ API Key Not Found")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    extraction_mode = st.radio(
        "Extraction Mode",
        ["Q&A Mode", "Auto Extract"],
        help="Q&A: Ask questions about invoice | Auto Extract: Get all data automatically"
    )
    
    st.markdown("---")
    st.markdown("**Groq Models Available:**")
    st.markdown("- llama-3.1-70b-versatile")
    st.markdown("- mixtral-8x7b-32768")
    st.markdown("- gemma2-9b-it")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Invoice")
    uploaded_file = st.file_uploader(
        "Choose an invoice image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the invoice"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Invoice", use_column_width=True)

with col2:
    st.subheader("🤖 AI Analysis")
    
    if extraction_mode == "Q&A Mode":
        input_text = st.text_input(
            "Ask a question about the invoice:",
            key="input",
            placeholder="e.g., What is the total amount?"
        )
        submit_button = st.button("🔍 Analyze Invoice", type="primary")
    else:
        submit_button = st.button("📊 Extract All Data", type="primary")

# Input prompt for the AI
input_prompt = """
You are an expert in understanding invoices across multiple languages.
You will receive extracted text from invoices and you need to answer 
questions based on the invoice data accurately. Provide clear, concise,
and accurate responses.
"""

# Handle submission
if submit_button and uploaded_file is not None:
    with st.spinner("🔄 Extracting text from invoice..."):
        # Extract text using OCR
        extracted_text = extract_text_from_image(uploaded_file)
    
    if extracted_text:
        # Show extracted text in expander
        with st.expander("📝 View Extracted Text"):
            st.text_area("OCR Output", extracted_text, height=200)
        
        if extraction_mode == "Q&A Mode":
            if input_text:
                with st.spinner("🤔 Analyzing with Groq AI..."):
                    response = get_groq_response(input_prompt, extracted_text, input_text)
                
                if response:
                    st.subheader("💡 Response")
                    st.success(response)
            else:
                st.warning("⚠️ Please enter a question about the invoice")
        
        else:  # Auto Extract Mode
            with st.spinner("📊 Extracting structured data..."):
                invoice_data = extract_invoice_data(extracted_text)
            
            if invoice_data:
                st.subheader("📋 Extracted Invoice Data")
                
                # Display as formatted data
                if "raw_response" not in invoice_data:
                    # Display in columns
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Invoice Number", invoice_data.get("invoice_number", "N/A"))
                        st.metric("Invoice Date", invoice_data.get("invoice_date", "N/A"))
                        st.metric("Vendor", invoice_data.get("vendor_name", "N/A"))
                    
                    with col_b:
                        st.metric("Customer", invoice_data.get("customer_name", "N/A"))
                        st.metric("Total Amount", f"{invoice_data.get('currency', '$')} {invoice_data.get('total_amount', 'N/A')}")
                    
                    # Line items if available
                    if "line_items" in invoice_data and invoice_data["line_items"]:
                        st.subheader("📦 Line Items")
                        st.json(invoice_data["line_items"])
                    
                    # Download JSON
                    st.download_button(
                        "⬇️ Download JSON",
                        data=json.dumps(invoice_data, indent=2),
                        file_name="invoice_data.json",
                        mime="application/json"
                    )
                else:
                    st.write(invoice_data["raw_response"])

elif submit_button:
    st.warning("⚠️ Please upload an invoice image first")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This app uses Tesseract OCR for text extraction and Groq API for AI analysis.
API key is loaded from `.env` file.
""")