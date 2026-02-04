# ==========================================
# File: cv_matcher.py
# Description: VacancyResumeMatcher class for cv analysis
# Author: @Olga492024
# ==========================================
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class VacancyResumeMatcher:
    """
    Система для матчинга резюме с вакансиями
    Использует векторный поиск через nomic-embed-text и комбинированный скоринг
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Инициализация модели эмбеддингов
        """
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.scaler = MinMaxScaler()

    def extract_text_from_docx(self, filepath: str) -> str:
        """
        Извлечение текста из DOCX файла
        """
        try:
            from docx import Document
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Ошибка при чтении {filepath}: {e}")
            return ""

    def load_resumes(self, cv_folder: str) -> Dict[int, str]:
        """
        Загрузка всех резюме из папки CV
        """
        resumes = {}
        cv_path = Path(cv_folder)

        for docx_file in sorted(cv_path.glob("*.docx")):
            try:
                cv_id = int(docx_file.stem)
                text = self.extract_text_from_docx(str(docx_file))
                if text.strip():
                    resumes[cv_id] = text
                    print(f"✓ Загружено резюме {cv_id}")
            except ValueError:
                continue

        print(f"\nВсего загружено резюме: {len(resumes)}")
        return resumes

    def load_vacancies(self, csv_path: str) -> Dict[int, Dict]:
        """
        Загрузка вакансий из CSV файла
        """
        vacancies = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vacancy_id = int(row['id'])
                vacancies[vacancy_id] = {
                    'title': row['job_title'],
                    'description': row['job_description'],
                    'uid': row['uid']
                }

        print(f"Загружено вакансий: {len(vacancies)}")
        return vacancies

    def extract_key_terms(self, text: str) -> List[str]:
        """
        Извлечение ключевых терминов из текста
        """
        skill_patterns = [
            r'\b(python|java|c\+\+|c#|javascript|typescript|php|ruby|go|rust|kotlin)\b',
            r'\b(sql|mysql|postgresql|oracle|mongodb|elasticsearch|redis)\b',
            r'\b(react|angular|vue|django|flask|spring|asp\.net|express)\b',
            r'\b(aws|azure|gcp|kubernetes|docker|jenkins|gitlab)\b',
            r'\b(git|svn|tfs|mercurial)\b',
            r'\b(agile|scrum|kanban|devops|ci/cd)\b',
            r'\b(rest|api|graphql|soap|microservices)\b',
            r'\b(linux|unix|windows|macos)\b',
            r'\b(html|css|xml|json|yaml)\b',
            r'\b(testing|unit test|integration test|qa|qc)\b',
        ]

        terms = []
        text_lower = text.lower()

        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                term = match.group(1)
                if term not in terms:
                    terms.append(term)

        return terms

    def calculate_skill_overlap(self, vacancy_text: str, resume_text: str) -> float:
        """
        Расчёт перекрытия навыков между вакансией и резюме (0-1)
        """
        vacancy_skills = set(self.extract_key_terms(vacancy_text))
        resume_skills = set(self.extract_key_terms(resume_text))

        if not vacancy_skills:
            return 0.5

        overlap = len(vacancy_skills.intersection(resume_skills))
        jaccard_similarity = overlap / len(vacancy_skills.union(resume_skills))

        return jaccard_similarity

    def calculate_text_length_match(self, vacancy_text: str, resume_text: str) -> float:
        """
        Метрика соответствия по длине текстов
        """
        vacancy_length = len(vacancy_text.split())
        resume_length = len(resume_text.split())

        if resume_length < vacancy_length * 0.3:
            return 0.5
        elif resume_length > vacancy_length * 2:
            return 0.7
        else:
            return 1.0

    def encode_texts(self, texts) -> np.ndarray:
        """
        Кодирование текстов в векторы с помощью модели эмбеддингов
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings

    def rank_vacancies_for_resume(self, resume_text: str,
                                   vacancies: Dict[int, Dict],
                                   vacancy_embeddings: Dict[int, np.ndarray],
                                   resume_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """
        Ранжирование вакансий для конкретного резюме
        """
        scores = []

        for vacancy_id, vacancy in vacancies.items():
            vacancy_text = vacancy['description']
            vacancy_embedding = vacancy_embeddings[vacancy_id]
            
            # 1. Косинусное сходство
            cosine_sim = cosine_similarity(
                [resume_embedding],
                [vacancy_embedding]
            )[0][0]
            
            # 2. Перекрытие навыков
            skill_overlap = self.calculate_skill_overlap(vacancy_text, resume_text)

            # 3. Соответствие по длине
            length_match = self.calculate_text_length_match(vacancy_text, resume_text)

            # Комбинированный скор
            combined_score = (
                0.60 * cosine_sim +
                0.25 * skill_overlap +
                0.15 * length_match
            )

            scores.append((vacancy_id, combined_score))

        # Сортировка по убыванию скора
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def calculate_ndcg(self, predicted_ranking: List[int],
                  ground_truth_ranking: List[int], k: int = 5) -> float:
        """
        Расчёт NDCG@k с учётом ПОРЯДКА ground truth
        """
        if not ground_truth_ranking or len(ground_truth_ranking) == 0:
            return 0.0

        # Создаём маппирование: вакансия → её идеальная позиция (релевантность)
        gt_positions = {vacancy_id: idx for idx, vacancy_id in enumerate(ground_truth_ranking)}

        # DCG - суммируем релевантность по позициям предсказания
        dcg = 0.0
        for i, vacancy_id in enumerate(predicted_ranking[:k]):
            if vacancy_id in gt_positions:
                # Релевантность = 1/(позиция в GT + 1)
                # Вакансия на позиции 0 в GT получает вес 1
                # Вакансия на позиции 1 в GT получает вес 1/2 и т.д.
                relevance = 1.0 / (gt_positions[vacancy_id] + 1)
                dcg += relevance / np.log2(i + 2)

        # IDCG - идеальный случай: все вакансии на своих позициях
        idcg = 0.0
        for i in range(min(k, len(gt_positions))):
            relevance = 1.0 / (i + 1)
            idcg += relevance / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    def evaluate_on_ground_truth(self, predictions: Dict[int, List[Tuple[int, float]]],
                                ground_truth: Dict[int, List[int]],
                                k: int = 5) -> Dict:
        """
        Оценка качества предсказаний с использованием ground truth аннотаций
        """
        ndcg_scores = []
        evaluated_count = 0

        for cv_id in sorted(predictions.keys()):
            if cv_id not in ground_truth:
                print(f"⚠ Резюме {cv_id}: нет ground truth")
                continue

            # Получаем ID вакансий из предсказаний (ранжированные)
            predicted_vacancy_ids = [pred[0] for pred in predictions[cv_id]]
            ground_truth_ranking = ground_truth[cv_id]

            # Проверим ground truth
            if not ground_truth_ranking or len(ground_truth_ranking) == 0:
                print(f"⚠ Резюме {cv_id}: пустой ground truth")
                continue

            # Рассчитываем NDCG
            ndcg = self.calculate_ndcg(predicted_vacancy_ids, ground_truth_ranking, k=k)
            ndcg_scores.append(ndcg)
            evaluated_count += 1

            if cv_id <= 5:
                print(f"  CV #{cv_id}: NDCG@{k}={ndcg:.4f}, GT={ground_truth_ranking}, Pred top 3={predicted_vacancy_ids[:3]}")

        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

        return {
            'avg_ndcg@5': avg_ndcg,
            'ndcg_scores': ndcg_scores,
            'count_evaluated': evaluated_count
        }