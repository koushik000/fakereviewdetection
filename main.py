import streamlit as st
import PyPDF2
import docx
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Functions to extract text from different file formats
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = [paragraph.text for paragraph in doc.paragraphs]
    return '\n'.join(text)

def extract_text(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file)
    elif file_extension == 'txt':
        return file.getvalue().decode('utf-8')
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_tokens)

def extract_skills(text, skills_list):
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_list:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills

def calculate_ats_score(resume_text, job_description, common_skills):
    # Preprocess texts
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(job_description)
    
    # Calculate keyword match score
    resume_skills = extract_skills(resume_text.lower(), common_skills)
    skill_match_ratio = len(resume_skills) / len(common_skills) if common_skills else 0
    
    # Calculate content similarity using cosine similarity
    vectorizer = CountVectorizer()
    try:
        count_matrix = vectorizer.fit_transform([processed_resume, processed_jd])
        similarity = cosine_similarity(count_matrix)[0][1]
    except:
        similarity = 0
    
    # Calculate final ATS score (you can adjust weights)
    keyword_weight = 0.6
    content_weight = 0.4
    
    final_score = (skill_match_ratio * keyword_weight) + (similarity * content_weight)
    final_score = min(final_score * 100, 100)  # Convert to percentage and cap at 100
    
    return final_score, resume_skills

def generate_feedback(resume_text, job_description, matched_skills, missing_skills, score):
    feedback = []
    
    # Overall assessment
    if score >= 80:
        feedback.append("✅ Your resume is well-optimized for this job description.")
    elif score >= 60:
        feedback.append("⚠️ Your resume is moderately optimized but could use some improvements.")
    else:
        feedback.append("❌ Your resume needs significant optimization for this job description.")
    
    # Skills assessment
    feedback.append(f"✅ Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")
    feedback.append(f"❌ Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")
    
    # Format checks
    if len(resume_text.split()) > 700:
        feedback.append("⚠️ Your resume might be too lengthy. Consider condensing it.")
    
    # Contact information check
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    
    if not re.search(email_pattern, resume_text, re.IGNORECASE):
        feedback.append("❌ Missing email address or format issue.")
    
    if not re.search(phone_pattern, resume_text):
        feedback.append("❌ Missing phone number or format issue.")
    
    return feedback

# Define common skills by industry (simplified)
INDUSTRY_SKILLS = {
    "Software Development": [
        "python", "java", "javascript", "html", "css", "react", "angular", "vue", "node.js", 
        "express", "django", "flask", "sql", "nosql", "mongodb", "aws", "azure", "gcp",
        "docker", "kubernetes", "ci/cd", "git", "agile", "scrum", "rest api", "testing",
        "debugging", "algorithms", "data structures", "oop", "functional programming"
    ],
    "Data Science": [
        "python", "r", "sql", "machine learning", "deep learning", "neural networks", "nlp",
        "computer vision", "statistics", "probability", "data visualization", "tableau",
        "power bi", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "big data", "hadoop", "spark", "regression", "classification", "clustering"
    ],
    "Marketing": [
        "digital marketing", "content marketing", "seo", "sem", "social media", "facebook ads",
        "google ads", "email marketing", "marketing automation", "analytics", "crm", "hubspot",
        "salesforce", "mailchimp", "campaign management", "market research", "brand management",
        "copywriting", "content strategy", "a/b testing", "conversion optimization"
    ],
    "Finance": [
        "financial analysis", "financial modeling", "accounting", "bookkeeping", "quickbooks",
        "excel", "forecasting", "budgeting", "taxation", "risk management", "investment",
        "banking", "portfolio management", "financial reporting", "audit", "compliance",
        "regulatory reporting", "sap", "oracle", "financial statements"
    ],
    "Healthcare": [
        "patient care", "medical records", "emr", "ehr", "epic", "cerner", "medical coding",
        "hipaa", "clinical", "diagnostic", "treatment", "medical terminology", "patient advocacy",
        "care coordination", "medicare", "medicaid", "insurance verification", "healthcare compliance",
        "medical billing", "telemedicine"
    ],
    "General": [
        "microsoft office", "excel", "word", "powerpoint", "outlook", "project management",
        "leadership", "communication", "teamwork", "problem solving", "critical thinking",
        "time management", "organization", "customer service", "research", "analysis", 
        "reporting", "presentation", "negotiation", "collaboration"
    ]
}

# Build the Streamlit app
def main():
    st.set_page_config(page_title="ATS Resume Scorer", layout="wide")
    
    st.title("ATS Resume Scoring System")
    st.markdown("""
    This app helps you understand how Applicant Tracking Systems (ATS) might score your resume 
    against a specific job description. Upload your resume and paste the job description to get started.
    """)
    
    # Layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Resume")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        
        industry = st.selectbox(
            "Select Industry (for specialized skill matching)",
            ["Software Development", "Data Science", "Marketing", "Finance", "Healthcare", "General"]
        )
        
        st.subheader("Job Description")
        job_description = st.text_area("Paste the job description here", height=300)
        
        additional_skills = st.text_input("Add any additional keywords to look for (comma-separated)")
        
    # Process the resume and job description
    if uploaded_file and job_description:
        with st.spinner("Analyzing your resume..."):
            # Extract text from resume
            resume_text = extract_text(uploaded_file)
            
            if resume_text:
                # Get skills list based on industry
                skills_list = INDUSTRY_SKILLS.get(industry, INDUSTRY_SKILLS["General"])
                
                # Add user-specified skills
                if additional_skills:
                    user_skills = [skill.strip().lower() for skill in additional_skills.split(",")]
                    skills_list.extend(user_skills)
                
                # Calculate ATS score
                score, matched_skills = calculate_ats_score(resume_text, job_description, skills_list)
                missing_skills = [skill for skill in skills_list if skill not in matched_skills]
                
                # Generate feedback
                feedback = generate_feedback(resume_text, job_description, matched_skills, missing_skills, score)
                
                with col2:
                    st.subheader("ATS Analysis Results")
                    
                    # Score gauge
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    ax.barh([0], [score], color='green' if score >= 70 else 'orange' if score >= 50 else 'red')
                    ax.barh([0], [100], color='lightgrey', left=0, alpha=0.3)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    ax.text(score/2, 0, f"{score:.1f}%", ha='center', va='center', color='white', fontweight='bold')
                    st.pyplot(fig)
                    
                    # Matched skills ratio
                    st.subheader("Skills Match")
                    st.write(f"**Matched:** {len(matched_skills)} out of {len(skills_list)} relevant skills")
                    
                    # Skill match visualization
                    match_ratio = len(matched_skills) / len(skills_list) if skills_list else 0
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    ax.barh([0], [match_ratio * 100], color='blue')
                    ax.barh([0], [100], color='lightgrey', left=0, alpha=0.3)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    ax.text(match_ratio * 50, 0, f"{match_ratio * 100:.1f}%", ha='center', va='center', color='white', fontweight='bold')
                    st.pyplot(fig)
                    
                    # Detailed feedback
                    st.subheader("Detailed Feedback")
                    for item in feedback:
                        st.markdown(item)
                    
                    # Skills tables
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.subheader("Matched Skills")
                        if matched_skills:
                            for skill in sorted(matched_skills):
                                st.markdown(f"✅ {skill}")
                        else:
                            st.write("No skills matched")
                            
                    with col_b:
                        st.subheader("Missing Skills")
                        if missing_skills:
                            for skill in sorted(missing_skills)[:10]:  # Show top 10 missing skills
                                st.markdown(f"❌ {skill}")
                            if len(missing_skills) > 10:
                                st.write(f"... and {len(missing_skills) - 10} more")
                        else:
                            st.write("No missing skills")
                    
                    # View extracted resume text
                    with st.expander("View Extracted Resume Text"):
                        st.text_area("", resume_text, height=200)

if __name__ == "__main__":
    main()