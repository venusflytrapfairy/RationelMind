# RationelMind AI - Research Intelligence Dashboard

## 1. Project Overview

RationelMind AI is a sophisticated web application designed to accelerate and deepen academic research. It transforms a set of related research papers (PDFs) into a dynamic, interactive intelligence dashboard. By leveraging the power of Google's Gemini AI, it moves beyond simple summarization to provide a multi-faceted analysis of the documents' core arguments, biases, and conceptual relationships.

The application is built for researchers, students, and analysts who need to quickly understand the landscape of a particular research topic, identify key conflicts between authors, and discover new pathways for investigation.

---

## 2. Core Features

-   **User-Defined Topic Analysis:** The user provides their core research topic, and all analyses are benchmarked against it.
-   **Relevance Scoring (Pie Chart):** Instantly visualizes how relevant each uploaded document is to the user's stated topic, providing a clear percentage breakdown.
-   **Narrative Conflict Graph:** Generates a dynamic network graph that maps out the conceptual conflict between papers. Instead of simple keywords, the graph uses descriptive phrases to explain each author's core argument, telling a visual story.
-   **In-Depth Paper Analysis:**
    -   **Summaries:** Concise summaries of each document's main points.
    -   **Bias Assessment:** Each paper is assessed for bias (Low, Medium, High). A **tooltip** on the bias flag provides a clear, evidence-based justification based on established academic criteria.
    -   **Construct Analysis:** Extracts and displays both the shared concepts used across all papers and the unique constructs exclusive to each one.
-   **Intelligence Modules:**
    -   **Reference Intelligence:** Scans the bibliographies of the uploaded papers to recommend a foundational or highly relevant cited work.
    -   **Multidisciplinary Connections:** Suggests connections between the core topic and other fields of study to spark new research ideas.

---

## 3. How Bias is Assessed

The AI is explicitly instructed to assess bias based on the following academic criteria:

1.  **Framing Bias:** Looks for consistently loaded or emotionally charged language that suggests a pre-determined conclusion.
2.  **Limited Scope / Selection Bias:** Flags papers that draw broad conclusions from an overly narrow data set without acknowledging the limitations.
3.  **Omission of Counter-Arguments:** Identifies when a paper fails to address or acknowledge well-known opposing viewpoints or alternative theories.

The justification for the assessment is available by hovering the mouse over the "BIAS" flag in the user interface.

---

## 4. Tech Stack

-   **Backend:** Python 3.x, Flask
-   **AI Engine:** Google Gemini AI (via `google-generativeai` library)
-   **PDF Parsing:** `PyMuPDF` (fitz)
-   **Frontend:** HTML5, CSS3, JavaScript (ES6)
-   **Visualization Libraries:**
    -   `vis.js` (for the narrative conflict network)
    -   `Chart.js` (for the relevance pie chart)
-   **Environment Management:** `python-dotenv` for API key management.

---

## 5. Setup and Installation

Follow these instructions to set up the project on your local machine.

### **Prerequisites:**
- Python 3.7+
- `pip` (Python package installer)

### **Step 1: Clone the Repository**
```bash
# This is a hypothetical command for when you host your code
git clone https://github.com/bonjour-paris/RationelMind.git
cd RationelMind
```

### **Step 2: Create a Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### **Step 3: Install Dependencies**
Create a `requirements.txt` file with the following content:

```text name=requirements.txt
flask
google-generativeai
python-dotenv
PyMuPDF
```

Then, install the packages using pip:
```bash
pip install -r requirements.txt
```

### **Step 4: Set Up Your API Key**
1.  Create a file named `.env` in the root directory of the project.
2.  Add your Google Gemini API key to this file as follows:

    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    Replace `"YOUR_API_KEY_HERE"` with your actual key.

---

## 6. How to Run the Application

1.  Ensure your virtual environment is activated.
2.  Run the Flask application from your terminal:
    ```bash
    python app.py
    ```
3.  You will see output indicating that the server is running, usually on `http://127.0.0.1:5000`.
    ```
     Gemini AI client configured successfully.
     * Serving Flask app 'app'
     * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
    ```
4.  Open your web browser and navigate to **`http://127.0.0.1:5000`**.

You can now use the RationelMind AI dashboard.
