import docx
import pandas as pd
import os

def read_docx (file_path):
    doc = docx.Document(file_path)
    full_text = []
    for p in doc.paragraphs:
        full_text.append(p.text)
        return "\n".join(full_text).strip()
    
def load_all_resumes(cv_folder = 'data/CV'):
    resumes = {}
    for fname in os.listdir(cv_folder):
        if fname.endswith('.docx'):
            rid = fname.replace('.docx', '')
            resumes[rid] = read_docx(os.path.join(cv_folder,fname))
    return resumes

def load_vacancies(file_path = 'data/5_vacancies.csv'):
    df = pd.read_csv(file_path)
    df = df[['id', 'job_description', 'job_title']]
    return df