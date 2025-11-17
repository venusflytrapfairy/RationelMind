import os
import google.generativeai as genai
from dotenv import load_dotenv

def run_gemini_test():
    """
    Loads API key, configures the Gemini client, and tests a simple prompt.
    """
    print("--- Starting Python Gemini Test ---")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from the environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ FAILURE: GEMINI_API_KEY not found in .env file.")
        return

    try:
        # Configure the generative AI client with your API key
        genai.configure(api_key=api_key)
        print("✔️ Gemini client configured.")

        # Select the generative model
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("✔️ Model 'gemini-pro' selected.")

        # Define the prompt
        prompt = "Explain how AI works in a few words"
        print(f"✔️ Sending prompt: '{prompt}'")

        # Generate content
        response = model.generate_content(prompt)

        # Print the response text
        print("\n✅ SUCCESS! Gemini responded:")
        print("---------------------------")
        print(response.text)
        print("---------------------------")

    except Exception as e:
        print(f"\n❌ FAILURE: An error occurred.")
        print(f"Error Details: {e}")

    finally:
        print("--- Test Finished ---")


# This line makes the script run when you execute it from the terminal
if __name__ == "__main__":
    run_gemini_test()