# ==========================================
# File: ranking.py
# Description: analysis skills similarity and ranking
# Author: @wavvybaby
# ==========================================
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


embedder = SentenceTransformer('all-MiniLM-L6-v2')

def skill_similarity(resume_skills, vacancy_skills):
    """
    Возвращает числовую метрику similarity
    """
    if not resume_skills or not vacancy_skills:
        return 0.0
    
    emb_res = embedder.encode(resume_skills, convert_to_tensor=True)
    emb_vac = embedder.encode(vacancy_skills, convert_to_tensor=True)

    sim_matrix = util.cos_sim(emb_res, emb_vac)
    
    return float(sim_matrix.mean())

def rank_resumes_for_vacancy(resume_skills_dict, vacancy_skills):
    scores = {}
    for rid, skills in resume_skills_dict.items():
        score = skill_similarity(skills, vacancy_skills)
        scores[rid] = score
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked