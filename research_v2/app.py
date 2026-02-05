import streamlit as st
from docx import Document
from src.preprocessing import load_vacancies
from src.extract_skills import extract_skills
from src.ranking import skill_similarity

st.title("AI Resume Matcher")

vacancies = load_vacancies()

st.sidebar.header("Вакансия")
vac_title = st.sidebar.selectbox(
    "Выберите вакансию",
    vacancies["job_title"].tolist()
)

vac_row = vacancies[vacancies["job_title"] == vac_title].iloc[0]
vac_skills = extract_skills(vac_row.job_description)

uploaded = st.file_uploader("Загрузите резюме (.docx)", type=["docx"])

if uploaded:
    doc = Document(uploaded)
    text = "\n".join([p.text for p in doc.paragraphs])

    resume_skills = extract_skills(text)

    similarity = skill_similarity(resume_skills, vac_skills)

    st.subheader("Процент соответствия навыков")
    st.metric(label="Match %", value=f"{similarity * 100:.2f}%")

    st.subheader("Извлечённые навыки из резюме")
    st.write(resume_skills)

    st.subheader("Навыки вакансии")
    st.write(vac_skills)