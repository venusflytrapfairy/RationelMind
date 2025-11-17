# app.py
import streamlit as st
import fitz
import json
import google.generativeai as genai
import os
import re
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Load environment ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Set it in your Streamlit secrets or .env file.")
else:
    genai.configure(api_key=api_key)

# --- Safety settings ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- PIPELINE_PROMPT ---
PIPELINE_PROMPT = """
You are a world-class AI research synthesizer. Your mission is to generate a deep analysis report with several intelligence components.

**CRITICAL INSTRUCTIONS & OUTPUT FORMAT:**
Return ONLY a single JSON object. Do not use markdown.

1. Extract Core Information: identify key constructs and authors.
2. Summarize & Assess Bias: summary + bias assessment.
3. Analyze Causal Contradiction: central thesis, stances by paper, graph edges.
4. Reference Intelligence: recommend one foundational cited paper.
5. Multidisciplinary Connections: identify 2-3 other academic fields.

**JSON OUTPUT FORMAT**:
{{
  "construct_analysis": {{
    "shared": [],
    "unique_by_paper": []
  }},
  "paper_summaries": [],
  "causal_contradiction": {{
    "central_thesis": "",
    "stances": [],
    "graph": {{
      "nodes": [],
      "edges": []
    }}
  }},
  "reference_intelligence": {{
    "recommendation_found": false,
    "recommended_paper_title": "",
    "justification": ""
  }},
  "multidisciplinary_connections": []
}}

**Papers to Analyze:**
---
{text}
---
"""

# --- JSON parsing helper ---
def clean_and_parse_json(response_text):
    match = re.search(r'\{[\s\S]*\}', response_text)
    if not match:
        raise ValueError(f"No valid JSON found in response. RAW (first 500 chars):\n{response_text[:500]}")
    return json.loads(match.group(0).strip())

# --- Streamlit UI ---
st.set_page_config(page_title="RationelMind AI", layout="wide")
st.title("RationelMind AI - Research Intelligence")
st.write("Upload 2–3 PDFs to generate a deep intelligence report.")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Generate Intelligence Report"):
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Please upload at least 2 PDFs.")
    else:
        with st.spinner("Analyzing papers..."):
            try:
                # --- Combine PDF texts ---
                combined_text = ""
                for file in uploaded_files:
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    paper_text = "".join(page.get_text() for page in doc)[:8000]  # limit per paper
                    combined_text += f"--- Paper: {file.name} ---\n{paper_text}\n\n"
                    doc.close()

                # --- Format prompt ---
                prompt = PIPELINE_PROMPT.format(text=combined_text)

                # --- Call Gemini ---
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(
                    prompt,
                    stream=False,
                    safety_settings=SAFETY_SETTINGS
                )

                # --- Parse JSON safely ---
                parsed_data = clean_and_parse_json(response.text)

                st.success("✅ Analysis complete!")

                # --- Render Shared Constructs ---
                st.header("1. Shared Constructs")
                for c in parsed_data.get('construct_analysis', {}).get('shared', []):
                    st.markdown(f"- {c}")

                # --- Render Unique Constructs ---
                st.header("2. Unique Constructs by Paper")
                for item in parsed_data.get('construct_analysis', {}).get('unique_by_paper', []):
                    filename = item.get("filename") or item.get("paper") or "Unknown Paper"
                    constructs = item.get("unique") or item.get("constructs", [])
                    st.markdown(f"**{filename}**")
                    for u in constructs:
                        st.markdown(f"- {u}")

                # --- Render Summaries & Bias ---
                st.header("3. Paper Summaries & Bias Assessment")
                for paper in parsed_data.get('paper_summaries', []):
                    filename = paper.get("filename") or paper.get("paper") or "Unknown Paper"
                    st.markdown(f"**{filename} ({paper.get('authors','Unknown')})**")
                    st.markdown(f"- Summary: {paper.get('summary','No summary provided')}")
                    bias = paper.get('bias_assessment', {})
                    st.markdown(f"- Bias: {bias.get('level','Unknown')} ({bias.get('justification','No justification')})")

                # --- Render Causal Contradiction ---
                st.header("4. Causal Contradiction / Stances")
                thesis = parsed_data.get('causal_contradiction', {}).get('central_thesis', 'Not provided')
                st.markdown(f"**Central Thesis:** {thesis}")
                for stance in parsed_data.get('causal_contradiction', {}).get('stances', []):
                    filename = stance.get("filename") or stance.get("paper") or "Unknown Paper"
                    st.markdown(f"- {filename} ({stance.get('authors','Unknown')}): {stance.get('stance','No stance')}")

                # --- Render Visual Graph ---
                st.header("5. Visual Conflict Graph")
                graph_html = f"""
                <div id="graph-container" style="width: 100%; height: 450px;"></div>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <script>
                    var nodes = new vis.DataSet({json.dumps(parsed_data.get('causal_contradiction', {}).get('graph', {}).get('nodes', []))});
                    var edges = new vis.DataSet({json.dumps(parsed_data.get('causal_contradiction', {}).get('graph', {}).get('edges', []))});
                    var container = document.getElementById('graph-container');
                    var data = {{ nodes: nodes, edges: edges }};
                    var options = {{
                        layout: {{ hierarchical: false }},
                        nodes: {{ borderWidth: 2, font: {{ size: 16 }} }},
                        edges: {{ font: {{ align: 'middle', size: 14 }}, arrows: 'to' }},
                        physics: {{ solver: 'barnesHut', barnesHut: {{ gravitationalConstant: -30000, centralGravity: 0.1, springLength: 300 }} }},
                        interaction: {{ hover: true }}
                    }};
                    new vis.Network(container, data, options);
                </script>
                """
                components.html(graph_html, height=500, scrolling=True)

                # --- Render Reference Intelligence ---
                st.header("6. Reference Intelligence")
                ref = parsed_data.get('reference_intelligence', {})
                if ref.get('recommendation_found', False):
                    st.markdown(f"**Recommended Paper:** {ref.get('recommended_paper_title','')}")
                    st.markdown(f"{ref.get('justification','')}")
                else:
                    st.markdown("No specific paper could be recommended from the references.")

                # --- Render Multidisciplinary Connections ---
                st.header("7. Multidisciplinary Connections")
                for c in parsed_data.get('multidisciplinary_connections', []):
                    st.markdown(f"- **{c.get('field','Unknown')}**: {c.get('connection','No connection provided')}")

            except Exception as e:
                st.error(f"❌ Analysis Failed: {e}")
