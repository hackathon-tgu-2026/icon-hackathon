from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)

def extract_skills(text, top_n=25):
    """
    Возвращает ключевые фразы для резюме и вакансий
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words='english',
        top_n= top_n
    )
    return [k[0] for k in keywords]