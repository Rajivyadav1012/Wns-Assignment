from groq import Groq
import httpx
import urllib3
import warnings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

api_key = "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"

print("Testing Groq API connection with SSL bypass...")

try:
    # Create HTTP client without SSL verification
    http_client = httpx.Client(verify=False)
    client = Groq(api_key=api_key, http_client=http_client)
    print("✓ Client created")
    
    print("Sending test message...")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
        max_tokens=50
    )
    print(f"✅ SUCCESS! Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()