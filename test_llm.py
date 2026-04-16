import os
from dotenv import load_dotenv
from config.llm_config import get_llm

# Load environment variables
load_dotenv()

# Test with Ollama
print("=== Testing with Ollama ===")
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = "llama3.2"  # or "mistral"

try:
    llm = get_llm()
    print(f"✓ LLM created: {type(llm).__name__}")
    
    # Test a simple prompt
    response = llm.invoke("Say 'Ollama is working!' in exactly 3 words")
    print(f"Response: {response.content}")
    print("✓ Ollama test passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*40 + "\n")

# Test with OpenAI (optional - only if you have an API key)
print("=== Testing with OpenAI (optional) ===")
os.environ["LLM_PROVIDER"] = "openai"

try:
    llm = get_llm()
    print(f"✓ LLM created: {type(llm).__name__}")
    print("✓ OpenAI configuration is valid")
    
except ValueError as e:
    print(f"Note: {e}")
    print("(This is expected if OPENAI_API_KEY is not set)")