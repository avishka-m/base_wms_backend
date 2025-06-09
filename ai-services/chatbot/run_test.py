import uvicorn
import os

# Set environment variables to skip prompts
# os.environ["LANGSMITH_TRACING"] = "false"
# os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"  # Replace with actual key if needed

if __name__ == "__main__":
    print("Starting WMS Chatbot API in test mode...")
    print("API will be available at http://127.0.0.1:8001")
    print("API Documentation: http://127.0.0.1:8001/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "main_new:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    ) 