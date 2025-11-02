
# **DigiKaksha (AI-Based Smart Classroom Management)**

**(The YOUTUBE link for the working of this project : https://youtu.be/s1tvwx_CCRs)**


DigiKaksha is a machine learning-based smart solution designed to assist and streamline tasks in educational institutions. It consists of three main components:

## **Face Detection-Based Attendance Monitoring**
Built using OpenCV, this module enables automatic attendance tracking through facial recognition. A camera setup (either inbuilt or external) is required to capture and identify students’ faces.

## **Student Performance Analysis**
The platform provides a graphical analysis of student performance by visualizing data such as marks and attendance. Graphs are generated using Matplotlib, offering insights into both individual and overall performance.

## **LangChain-Based Study Material Chatbot**
Utilizing LangChain, a retriever-augmented generation (RAG) model, and the Google Gemini API, the chatbot helps students comprehend their notes or study materials. It supports querying uploaded PDFs and generates responses based on their content. The ingestion pipeline uses Docling to convert PDFs/Office docs to Markdown, performs header-aware and recursive chunking, then embeds with the Hugging Face model `Qwen/Qwen3-Embedding-0.6B` into a Chroma vector store (cosine similarity, persisted in `./VectorDB/`). The embedding model can be overridden via the `EMBED_MODEL_ID` environment variable; if you change it, delete `./VectorDB/` and re-upload to re-embed.

---

## **Key Technologies Used**
- **LangChain and RAG:** To retrieve and generate answers based on study materials.
- **OpenCV:** For face detection and attendance monitoring.
- **Matplotlib:** For creating graphs and analyzing student performance.
- **MongoDB:** As the database backend to store student data.

---

## **Key Components**

### 1. **Face Detection for Attendance:**
- Uses face recognition to mark attendance automatically.
- Stores known student faces and compares them with real-time input to log attendance.
- The attendance is marked with the time in a CSV file.

### 2. **Student Data Analysis:**
- Analyzes student marks and attendance across subjects.
- Provides bar and pie charts for visual insights, including individual student performance or an overall comparison of class averages.

### 3. **LangChain-Powered Chatbot:**
- Loads PDF study materials and splits text into manageable chunks using recursive character text splitting.
- Embeds the text using Hugging Face models and stores embeddings in Chroma.
- Provides context-aware answers to students' questions via a chatbot interface.

---

## **Requirements**
- Install necessary modules using `pip install -r requirements.txt`.
- Python 3.11.X or below is recommended, as Python 3.12.X may cause compatibility issues.
- Ensure a camera setup is available for face recognition.

### **API Keys Required:**
- **LANGCHAIN_TRACING_V2** for LangChain tracing. (LangChain API key)
- **GOOGLE_API_KEY** for Google Gemini model integration. (Gemini API key)

---

## **Setup Instructions**
1. Install dependencies via the `requirements.txt` file.
2. Ensure you have a valid `LANGCHAIN_TRACING_V2` and `GOOGLE_API_KEY` to use LangChain and Google APIs.
3. Set up MongoDB for storing and retrieving student data.
4. Run the app using FastAPI, accessible at `http://localhost:8000/`.

---

## **Additional Notes**
- Face detection and attendance logging is handled via `face.py`.
- Uploaded study materials (PDFs) are processed, and a retriever system is used to answer queries based on the study content.
- Graphs for performance analysis can be viewed via the web interface, allowing individual or class-wide analysis.

