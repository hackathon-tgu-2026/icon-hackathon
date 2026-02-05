import ast
from src.preprocessing import load_all_resumes, load_vacancies
from src.extract_skills import extract_skills
from src.ranking import rank_resumes_for_vacancy
from src.metrics import compute_metrics

def load_annotations(path='data/annotations-for-the-first-30-vacancies.txt'):
    text = open(path,'r').read()
    a1 = ast.literal_eval(text.split('ANNOTATOR_1_RANKINGS')[1].split('\n')[0])
    a2 = ast.literal_eval(text.split('ANNOTATOR_2_RANKINGS')[1].split('\n')[0])

    avg_rankings = []
    for r1, r2 in zip(a1,a2):
        avg = [(x+y)/2 for x,y in zip(r1,r2)]
        avg_rankings.append(avg)
    
    return avg_rankings

def train_and_evaluate():
    resumes = load_all_resumes()
    vacancies = load_vacancies()
    annot = load_annotations()

    resume_skills_dict = {rid: extract_skills(text) for rid, text in resumes.items()}
    vacancy_skills_dict = {row.id: extract_skills(row.job_description) for _,row in vacancies.iterrows()}

    all_results = []

    for i, row in vacancies.iterrows():
        vid = row.id
        vac_sk = vacancy_skills_dict[vid]

        ranked = rank_resumes_for_vacancy(resume_skills_dict, vac_sk)

        model_order = [int(r[0]) for r in ranked if int(r[0]) <= 30][:5]
        hr_rank = annot[i]

        ndcg, spearman = compute_metrics(hr_rank, model_order)
        all_results.append((ndcg,spearman))

    print('РЕЗУЛЬТАТЫ МОДЕЛИ')
    for idx, r in enumerate(all_results):
        print(f'Вакансия {vacancies.iloc[idx].job_title}')
        print(f'--- NDCG {r[0]:.4f}')
        print(f'--- Spearman {r[1]:.4f}\n')

    avg_ndcg = sum(r[0] for r in all_results) / len(all_results)
    avg_spear = sum(r[1] for r in all_results) / len(all_results)

    print('СРЕДНИЕ МЕТРИКИ')
    print(f'Средний NDCG {avg_ndcg:.4f}')
    print(f'Средний Spearman {avg_spear:.4f}')

if __name__ == '__main__':
    train_and_evaluate()