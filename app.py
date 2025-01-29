import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai
from functools import lru_cache
from time import sleep

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ATS Analyzer Class
class ATSAnalyzer:
    @staticmethod
    # Removed caching for uploaded_file

    def extract_text_from_pdf(uploaded_file):
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return None

    @staticmethod
    def analyze_match_percentage(resume_text, job_description):
        # Vectorize text using CountVectorizer
        vectorizer = CountVectorizer().fit_transform([resume_text, job_description])
        vectors = vectorizer.toarray()

        # Compute cosine similarity
        cosine_sim = cosine_similarity(vectors)
        match_percentage = round(cosine_sim[0][1] * 100, 2)  # Percentage match
        return match_percentage

    @staticmethod
    def find_keywords(resume_text, job_description):
        # Extract words from job description and check for presence in resume
        job_keywords = set(job_description.lower().split())
        resume_keywords = set(resume_text.lower().split())

        matching_keywords = job_keywords.intersection(resume_keywords)
        missing_keywords = job_keywords.difference(resume_keywords)

        return matching_keywords, missing_keywords

    @staticmethod
    def get_gemini_response(prompt, retries=3):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([prompt])
            return response.text
        except genai.errors.RateLimitError:
            if retries > 0:
                st.warning("Rate limit reached. Retrying...")
                sleep(2)
                return ATSAnalyzer.get_gemini_response(prompt, retries - 1)
            else:
                st.error("API rate limit exceeded. Please try again later.")
                return None
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None


# Main Application
def main():
    st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")

    # Custom CSS for UI styling
    st.markdown("""
        <style>
        .stButton>button { width: 100%; background-color: #0066cc; color: white; }
        .stButton>button:hover { background-color: #0052a3; }
        .success-message { padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; color: #155724; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("📄 ATS Resume Analyzer")
    st.markdown("""
        This tool helps you analyze your resume against job descriptions using AI. 
        Upload your resume and paste the job description to:
        - Get a match percentage with job requirements
        - Identify missing keywords and areas for improvement
    """)

    # Layout Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area("Paste the job description here:", height=200, placeholder="Paste the complete job description here...")

    with col2:
        st.subheader("📎 Resume Upload")
        uploaded_file = st.file_uploader("Upload your resume (PDF format):", type=["pdf"], help="Please ensure your resume is in PDF format")
        if uploaded_file:
            st.markdown('<p class="success-message">✅ PDF uploaded successfully!</p>', unsafe_allow_html=True)

    if uploaded_file and job_description:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume... Please wait"):
                resume_text = ATSAnalyzer.extract_text_from_pdf(uploaded_file)  # No caching

                if resume_text:
                    # Calculate match percentage
                    match_percentage = ATSAnalyzer.analyze_match_percentage(resume_text, job_description)

                    # Find matching and missing keywords
                    matching_keywords, missing_keywords = ATSAnalyzer.find_keywords(resume_text, job_description)

                    # Display Results
                    st.markdown("### Analysis Results")
                    st.write(f"**Match Percentage:** {match_percentage}%")
                    st.write("**Matching Keywords:**", ", ".join(matching_keywords) if matching_keywords else "None")
                    st.write("**Missing Keywords:**", ", ".join(missing_keywords) if missing_keywords else "None")

                    # Optional: Use Gemini AI for recommendations
                    prompt = f"""
                    Analyze the following resume for improvements based on the job description:
                    Resume:
                    {resume_text}
                    Job Description:
                    {job_description}
                    Provide recommendations to improve alignment with the job role.
                    """
                    gemini_response = ATSAnalyzer.get_gemini_response(prompt)
                    if gemini_response:
                        st.markdown("### AI Recommendations")
                        st.markdown(gemini_response)

                    # Export Analysis
                    analysis_report = f"""
                    Match Percentage: {match_percentage}%
                    Matching Keywords: {", ".join(matching_keywords)}
                    Missing Keywords: {", ".join(missing_keywords)}
                    AI Recommendations: {gemini_response if gemini_response else 'No recommendations generated.'}
                    """
                    st.download_button(
                        label="📥 Export Analysis",
                        data=analysis_report,
                        file_name="resume_analysis.txt",
                        mime="text/plain"
                    )
    else:
        st.info("👆 Please upload your resume and provide the job description to begin the analysis.")

    # Footer
    st.markdown("---")
    st.markdown(
        "Made with ❤️ by Your Company | This tool uses AI to analyze resumes but should be used as one of many factors in your job application process."
    )


if __name__ == "__main__":
    main()
