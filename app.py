import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI use

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request as StarletteRequest
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key='your_secret_key')

templates = Jinja2Templates(directory="templates")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables from .env
load_dotenv()

# Sample user credentials
users = {
    '1DS22CS100': 'Gx#8dLpM2a',
    '1DS22CD034': 'password2',
    '1DS22CS102': 'Pm2aGx#8dL'
}

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Adjust the connection string if needed
db = client['education']  # Database name
collection = db['students']  # Collection name

# Initialize LangChain components, load API Keys from environment
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")

embeddings_model = None
path_db = "./VectorDB"
llm = None

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant. Use the provided context to answer the user's question. If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
    ("human", "Question: {question}")
])

retriever = None

def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    return embeddings_model

def get_student_data():
    students_data = list(collection.find({}, {'_id': 0}))  # Exclude the MongoDB default _id field
    return pd.json_normalize(students_data)

def format_docs(docs):
    return "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else doc for doc in docs)

def retrieve_and_format(query):
    # In LangChain 1.x, retrievers are Runnables; use .invoke instead of get_relevant_documents
    docs = retriever.invoke(query)
    return format_docs(docs)

def get_llm():
    global llm
    if llm is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Please set it in app.py before using the chatbot.")
        # Initialize Google Gemini LLM lazily
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    return llm

def get_rag_chain():
    if retriever is None:
        raise RuntimeError("Vector store not initialized. Upload PDFs at /upload first.")
    return (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: StarletteRequest):
    if 'username' not in request.session:
        return RedirectResponse(url='/login')
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.post("/token")
async def login_for_access_token(request: StarletteRequest, form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username in users and users[form_data.username] == form_data.password:
        request.session['username'] = form_data.username
        return {"access_token": form_data.username, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: StarletteRequest):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: StarletteRequest, username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        request.session['username'] = username
        return RedirectResponse(url='/', status_code=status.HTTP_302_FOUND)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})

@app.get("/logout")
async def logout(request: StarletteRequest):
    request.session.pop('username', None)
    return RedirectResponse(url='/login')

@app.get("/run-face-recognition")
async def run_face_recognition():
    subprocess.call(['python', 'face.py'])
    return RedirectResponse(url='/')

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: StarletteRequest):
    return templates.TemplateResponse('upload_file.html', {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: StarletteRequest, file: UploadFile = File(...)):
    if not file.filename:
        return RedirectResponse(url='/upload')
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    process_file(file_path)
    return RedirectResponse(url='/ask', status_code=status.HTTP_302_FOUND)

def process_file(file_path):
    """Process an uploaded file into the vector store using Docling for conversion and Markdown-aware chunking."""
    global retriever

    # Convert the uploaded file to Markdown using Docling
    converter = DocumentConverter()
    result = converter.convert(file_path)
    markdown = result.document.export_to_markdown()
    docs = [Document(page_content=markdown, metadata={"source": os.path.basename(file_path), "parser": "docling"})]

    # Header-aware splitting for better semantic chunks
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    split_docs: list[Document] = []
    for d in docs:
        split_docs.extend(header_splitter.split_text(d.page_content))

    # Further split long sections into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(split_docs)

    vectorstore = Chroma.from_documents(documents=chunks, embedding=get_embeddings_model(), persist_directory=path_db)
    retriever = vectorstore.as_retriever()

@app.get("/ask", response_class=HTMLResponse)
async def ask_page_get(request: StarletteRequest):
    return templates.TemplateResponse('ask.html', {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_page_post(request: StarletteRequest, question: str = Form(...)):
    if question:
        try:
            response = get_rag_chain().invoke(question)
        except RuntimeError as e:
            return templates.TemplateResponse('ask.html', {"request": request, "question": question, "error": str(e)})
        return templates.TemplateResponse('ask.html', {"request": request, "question": question, "response": response})
    return templates.TemplateResponse('ask.html', {"request": request})

@app.post("/ask_json")
async def ask_json(request: StarletteRequest, question: str = Form(...)):
    if question:
        try:
            response = get_rag_chain().invoke(question)
        except RuntimeError as e:
            return JSONResponse(content={'error': str(e)}, status_code=400)
        return JSONResponse(content={'response': response})
    return JSONResponse(content={'response': 'No question provided'})

@app.get("/graph_entry", response_class=HTMLResponse)
async def graph_entry(request: StarletteRequest):
    return templates.TemplateResponse('graph_entry.html', {"request": request})

@app.post("/analyze_individual", response_class=HTMLResponse)
async def analyze_individual(request: StarletteRequest, usn: str = Form(...)):
    df = get_student_data()
    print(df.columns)
    if 'usn' not in df.columns:
        raise HTTPException(status_code=400, detail="No 'usn' field in data.")

    student = df[df['usn'] == usn]

    if student.empty:
        raise HTTPException(status_code=404, detail="No student found with this USN.")

    student = student.iloc[0]
    subjects = ["MATHS", "IDS", "DBMS", "DAA", "CN", "MONGODB"]

    # Prepare results
    results = f"Analyzing student: {student['name']} (USN: {student['usn']})\n"
    results += f"Branch: {student['branch']}, Age: {student['age']}\n"
    results += f"Total Marks: {student['total_marks']}, Total Attendance Percentage: {student['total_attendance_percentage']}\n"
    
    # Subject-wise Marks
    marks = student[['marks.' + sub for sub in subjects]]
    results += f"Marks: {marks.to_dict()}\n"

    # Subject-wise Attendance
    attendance = student[['attendance_percentage.' + sub for sub in subjects]]
    results += f"Attendance: {attendance.to_dict()}\n"

    # Comparison with Average Marks
    average_marks = df[["marks." + sub for sub in subjects]].mean()
    results += f"Average Marks: {average_marks.to_dict()}\n"

    # Generate plots
    marks_image = f'{usn}_marks.png'
    attendance_image = f'{usn}_attendance.png'
    marks_pie_image = f'{usn}_marks_pie.png'
    attendance_pie_image = f'{usn}_attendance_pie.png'

    # Bar Plot for Marks
    plt.figure(figsize=(10, 5))
    plt.bar(subjects, marks, color='skyblue')
    plt.xlabel('Subjects')
    plt.ylabel('Marks')
    plt.title(f'Marks of {student["name"]}')
    plt.savefig(f'./static/{marks_image}')
    plt.close()

    # Pie Chart for Marks
    plt.figure(figsize=(8, 8))
    plt.pie(marks, labels=subjects, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(subjects))))
    plt.title(f'Marks Distribution for {student["name"]}')
    plt.savefig(f'./static/{marks_pie_image}')
    plt.close()

    # Bar Plot for Attendance
    plt.figure(figsize=(10, 5))
    plt.bar(subjects, attendance, color='lightgreen')
    plt.xlabel('Subjects')
    plt.ylabel('Attendance Percentage')
    plt.title(f'Attendance Percentage of {student["name"]}')
    plt.savefig(f'./static/{attendance_image}')
    plt.close()

    # Pie Chart for Attendance
    plt.figure(figsize=(8, 8))
    plt.pie(attendance, labels=subjects, autopct='%1.1f%%', colors=plt.cm.Paired(range(len(subjects))))
    plt.title(f'Attendance Distribution for {student["name"]}')
    plt.savefig(f'./static/{attendance_pie_image}')
    plt.close()

    individual = {
        'marks_image': marks_image,
        'attendance_image': attendance_image,
        'marks_pie_image': marks_pie_image,
        'attendance_pie_image': attendance_pie_image
    }

    return templates.TemplateResponse('graph_details.html', {"request": request, "results": results, "individual": individual, "overall": None})

@app.post("/analyze_all", response_class=HTMLResponse)
async def analyze_all(request: StarletteRequest):
    df = get_student_data()
    # Overall Performance Analysis
    subjects = ["MATHS", "IDS", "DBMS", "DAA", "CN", "MONGODB"]
    average_marks = df[["marks." + sub for sub in subjects]].mean()
    average_attendance = df[["attendance_percentage." + sub for sub in subjects]].mean()

    # Prepare results for all students
    results = "Overall Performance Analysis:\n"
    results += f"Average Marks: {average_marks.to_dict()}\n"
    results += f"Average Attendance: {average_attendance.to_dict()}\n"

    # Histogram of Total Marks
    total_marks_image = 'distribution_total_marks.png'
    plt.figure(figsize=(10, 5))
    plt.hist(df['total_marks'], bins=10, color='purple', alpha=0.7)
    plt.xlabel('Total Marks')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Total Marks')
    plt.savefig(f'./static/{total_marks_image}')
    plt.close()

    # Histogram of Total Attendance Percentage
    total_attendance_image = 'distribution_total_attendance.png'
    plt.figure(figsize=(10, 5))
    plt.hist(df['total_attendance_percentage'], bins=10, color='orange', alpha=0.7)
    plt.xlabel('Total Attendance Percentage')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Total Attendance Percentage')
    plt.savefig(f'./static/{total_attendance_image}')
    plt.close()

    overall = {
        'total_marks_image': total_marks_image,
        'total_attendance_image': total_attendance_image
    }

    return templates.TemplateResponse('graph_details.html', {"request": request, "results": results, "overall": overall, "individual": None})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
