import os
import fitz
import json
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import traceback
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Basic App Setup ---
app = Flask(__name__)
load_dotenv()

# --- Configure Gemini AI Client ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("✅ Gemini AI client configured successfully.")
except Exception as e:
    print(f"❌ CRITICAL: Failed to configure Gemini AI. {e}")

# --- THE ULTIMATE "INTELLIGENCE" PROMPT (v5) ---
PIPELINE_PROMPT = """
You are a world-class AI research synthesizer. Your mission is to generate a deep analysis report with several intelligence components.

**CRITICAL INSTRUCTIONS & OUTPUT FORMAT:**
Follow these steps and STRICTLY return ONLY a single JSON object. Do not use markdown.

1.  **Extract Core Information:** For each paper, identify the key constructs (shared and unique) and the primary author(s).
2.  **Summarize & Assess Bias:** For each paper, write a summary and perform a bias assessment (Low/Medium/High) based on academic criteria like framing, scope, and counter-arguments.
3.  **Analyze Causal Contradiction:**
    -   **Central Thesis:** State the core conflict as a sharp, single-sentence question.
    -   **Stances:** For each paper, provide its specific stance on the thesis, ATTRIBUTING the argument to the identified author(s). E.g., "In [Filename], Author X argues that...".
4.  **Generate a CLEAN Conflict Graph:**
    -   Create nodes for each paper and ONE for the central thesis.
    -   Create edges from each paper to the thesis node.
    -   The edge `label` MUST be a very short keyword phrase (1-2 words MAX) for visual clarity.
5.  **Reference Intelligence (New):**
    -   Scan the bibliographies OF THE PROVIDED PAPERS ONLY.
    -   Identify ONE paper title FROM WITHIN THOSE REFERENCES that appears most foundational or relevant to the core topic.
    -   Provide the title and a justification for why it's a recommended read. If no clear recommendation can be made, state that.
6.  **Multidisciplinary Connections (New):**
    -   Identify 2-3 other academic fields (e.g., Sociology, Computer Science, Economics) that the core topic connects to, and provide a brief explanation for each connection.

**JSON OUTPUT FORMAT:**
{{
  "construct_analysis": {{
    "shared": ["Shared Construct 1"],
    "unique_by_paper": [ {{"filename": "Filename 1", "unique": ["Unique Construct A"]}} ]
  }},
  "paper_summaries": [
    {{
      "filename": "Filename 1",
      "authors": "Author A, Author B",
      "summary": "This paper's summary.",
      "bias_assessment": {{"level": "Low", "justification": "Justification."}}
    }}
  ],
  "causal_contradiction": {{
    "central_thesis": "The core conflict, as a question.",
    "stances": [
      {{"filename": "Filename 1", "authors": "Author A", "stance": "Author A argues that..."}}
    ],
    "graph": {{
      "nodes": [
        {{"id": "paper_1", "label": "Filename 1", "shape": "box"}},
        {{"id": "thesis_node", "label": "Central Thesis Name", "shape": "ellipse", "size": 25}}
      ],
      "edges": [
        {{ "from": "paper_1", "to": "thesis_node", "label": "Keywords", "relationship_type": "disagreement"}}
      ]
    }}
  }},
  "reference_intelligence": {{
    "recommendation_found": true,
    "recommended_paper_title": "Title of the recommended cited paper.",
    "justification": "Why this paper seems more relevant or foundational."
  }},
  "multidisciplinary_connections": [
    {{"field": "Sociology", "connection": "Explanation of the connection."}}
  ]
}}

**Papers to Analyze:**
---
{text}
---
"""

SAFETY_SETTINGS = { HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, }

def clean_and_parse_json(response_text):
    match = re.search(r'\{[\s\S]*\}', response_text)
    if not match: raise ValueError(f"No valid JSON in Gemini response. RAW: {response_text}")
    return json.loads(match.group(0).strip())

@app.route('/analyze', methods=['POST'])
def analyze_documents():
    print("\nBackend: Received intelligence analysis request...")
    files = request.files.getlist('papers')
    if not files or len(files) < 2: return jsonify({"error": "Please upload 2 or 3 PDF files."}), 400

    try:
        combined_text = ""
        for file in files:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            combined_text += f"--- Paper: {file.filename} ---\n{text[:8000]}\n\n"

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = PIPELINE_PROMPT.format(text=combined_text)
        
        print("Backend: Sending intelligence prompt to Gemini...")
        response = model.generate_content(prompt, stream=False, safety_settings=SAFETY_SETTINGS)

        parsed_data = clean_and_parse_json(response.text)
        print("✅ Backend: Successfully returned intelligence report.")
        return jsonify(parsed_data)

    except Exception as e:
        print(f"--- BACKEND CRASH ---"); traceback.print_exc(); print(f"---------------------")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)