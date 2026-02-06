# ==========================================
# File: main.py
# Description: TG bot for analise CV and vacancies
# Author: @koleslena
# ==========================================
import os
import requests

from docx import Document
from io import BytesIO

import telebot
from telebot.types import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, KeyboardButton
import datetime
import logging

from cv_matcher import VacancyResumeMatcher

from extract_skills import extract_skills
from ranking import skill_similarity

logger = telebot.logger

formatter = '[%(asctime)s] %(levelname)8s --- %(message)s (%(filename)s:%(lineno)s)'
logging.basicConfig(
    filename=f'iconi_bot-from-{datetime.datetime.now().date()}.log',
    filemode='w',
    format=formatter,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.WARNING
)

RANK = 3
SAVE_FILES = False

if SAVE_FILES:
    DOWNLOAD_FOLDER = 'downloads'
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

VACANCIES_CSV = "./tg_bot/5_vacancies.csv"

try:
    cv_matcher = VacancyResumeMatcher()
    vacancies = cv_matcher.load_vacancies(VACANCIES_CSV)
    all_vacancy_texts = [vacancies[vid]['description'] for vid in sorted(vacancies.keys())]
    vacancy_ids = sorted(vacancies.keys())
    vacancy_embeddings_list = cv_matcher.encode_texts(all_vacancy_texts)
    vacancy_embeddings = {vid: vacancy_embeddings_list[i] for i, vid in enumerate(vacancy_ids)}
except Exception as e:
    logger.error(e)

TOKEN = os.environ.get("iconi_bot_token")

bot = telebot.TeleBot(TOKEN, parse_mode='HTML')

FIND_VACANCIES = "Find vacancies"
SHOW_MATCH = "Show match"

ACTIVE_STATE_INIT = {'mode': FIND_VACANCIES, 'id': None}
ACTIVE_STATE = {}


def read_docx(message, docx_file):
    try:
        cv_id = int(docx_file.stem)
        text = cv_matcher.extract_text_from_docx(str(docx_file))
        if text.strip():
            print(f"✓ Загружено резюме {cv_id}")
            return cv_id, text
    except ValueError:
        msg = bot.send_message(message, 'Что-то пошло не так. Попробуйте еще раз.')

def get_ranks(cv_id, resume_text):
    resume_embedding = cv_matcher.encode_texts(resume_text)
    
    # Ранжируем вакансии для этого резюме
    ranked = cv_matcher.rank_vacancies_for_resume(resume_text, vacancies, 
                                                  vacancy_embeddings, resume_embedding)

    return ranked[:RANK]

# Handle '/admin'
@bot.message_handler(commands=['admin'])
def handle_admin(message):
    pass

def gen_main_menu():
    markup = ReplyKeyboardMarkup(True, False)
    markup.add(KeyboardButton(FIND_VACANCIES))
    markup.add(KeyboardButton(SHOW_MATCH))
    return markup

def get_active_state(user_id):
    return ACTIVE_STATE.get(user_id, ACTIVE_STATE_INIT)

# Handle '/cancel'
@bot.message_handler(commands=['cancel'])
def send_cancel(message):
    get_active_state(message.from_user.id)['mode'] = FIND_VACANCIES
    get_active_state(message.from_user.id)['id'] = None
    markup = ReplyKeyboardRemove()
    msg = bot.send_message(message.from_user.id, f"""\
        Отмена \
        """, reply_markup=markup)

# Handle '/start'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    msg = bot.send_message(message.from_user.id, f"""\
        Привет, <i>{message.from_user.first_name}</i>. Я HRBot. \
        \n\nЯ помогу тебе с поиском вакансий. \
        \n\nЖми на кнопку {FIND_VACANCIES} чтобы подобрать топ-{RANK} вакансий по резюме, \
        \n\nили {SHOW_MATCH} чтобы посмотреть совпадения по конкретной вакансии. \
        \n\nПомощь /help. \
        """, reply_markup=gen_main_menu())

@bot.message_handler(func=lambda message: message.text == SHOW_MATCH)
def process_ask_question_match(message):
    get_active_state(message.from_user.id)['mode'] = SHOW_MATCH
    vacs_text = ""
    for vac_id, vac in vacancies.items():
        vacs_text += f"\n\n{vac_id}. {vac['title']}"
    
    msg = bot.send_message(message.from_user.id, f"""\
        \n\nВыбери вакансию, отправь мне только ее номер {vacs_text}  \
        """)

@bot.message_handler(func=lambda message: message.text == FIND_VACANCIES)
def process_ask_question(message):
    get_active_state(message.from_user.id)['mode'] = FIND_VACANCIES
    msg = bot.send_message(message.from_user.id, f"""\
        \n\nПришли мне docx файл резюме или краткий текст резюме одним сообщением \
        """)

# Handle '/help'
@bot.message_handler(commands=['help'])
def send_help(message):
    msg = bot.send_message(message.from_user.id, f"""\
        Привет, <i>{message.from_user.first_name}</i>. Я HRBot. \
        \n\nЯ помогу тебе с поиском вакансий. \
        \n\nЖми на кнопку {FIND_VACANCIES} чтобы подобрать топ-{RANK} вакансий по резюме, \
        \n\nили {SHOW_MATCH} чтобы посмотреть совпадения по конкретной вакансии. \       
        """, reply_markup=gen_main_menu())


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    try:
        pass
    except:
        msg = bot.send_message(call.from_user.id, 'Что-то пошло не так. Попробуйте еще раз.')

@bot.message_handler(content_types=['document'])
def handle_document(message):
    if message.document.file_name.endswith('.docx'):
        try:
            # Get file ID
            file_id = message.document.file_id
            # Get file information (file_path)
            file_info = bot.get_file(file_id)
            file_path = file_info.file_path

            # Construct the download URL
            download_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
            
            # Download the file
            response = requests.get(download_url)
            if response.status_code == 200:
                # Save the file to the local directory
                if SAVE_FILES:
                    save_path = os.path.join(DOWNLOAD_FOLDER, message.document.file_name)
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    cv_id, resume_text = read_docx(message, save_path)
                else:
                    docx_content = BytesIO(response.content)

                    # Open the document using python-docx
                    document = Document(docx_content)

                    # Extract all text, paragraph by paragraph
                    full_text = []
                    for paragraph in document.paragraphs:
                        full_text.append(paragraph.text)

                    # Join the paragraphs with newlines
                    extracted_text = '\n'.join(full_text)

                    cv_id, resume_text = file_id, extracted_text
                
                bot.reply_to(message, f"Получил файл, проверяю...")
                
                state = get_active_state(message.from_user.id)
                if state['mode'] == FIND_VACANCIES:
                    result = get_ranks(cv_id, resume_text)
                    build_answer(message, result, resume_text)
                elif state['mode'] == SHOW_MATCH and state['id']:
                    show_match(message, resume_text)
                else:
                    bot.reply_to(message, 'Что-то пошло не так. Попробуйте еще раз.')
            else:
                bot.reply_to(message, 'Что-то пошло не так. Попробуйте еще раз.')
        except Exception as e:
            print(e)
            # bot.reply_to(message, f"Ошибка: {e}")
    else:
        bot.reply_to(message, 'Только .docx файлы.')

def answer(message):
    state = get_active_state(message.from_user.id)
    if state['mode'] == FIND_VACANCIES:
        result = get_ranks(1, message.text)
        build_answer(message, result, message.text)
    elif state['mode'] == SHOW_MATCH and str.isdigit(message.text) and int(message.text) in vacancies.keys():
        state['id'] = int(message.text)
        bot.send_message(message.from_user.id, f"""\
            \n\nВыбрана вакансия {state['id']}. Теперь пришли мне docx файл резюме или краткий текст резюме одним сообщением \
            """)
    elif state['mode'] == SHOW_MATCH and not str.isdigit(message.text) and state['id']:
        bot.reply_to(message, f"Выбраная вакансия {state['id']}. Получил резюме, проверяю...")
        show_match(message, message.text)
    else: 
        bot.reply_to(message, 'Что-то пошло не так. Попробуйте еще раз.')

def show_match(message, resume_text):
    vac_id = get_active_state(message.from_user.id)['id']
    vacancy = vacancies[vac_id]

    vac_skills = extract_skills(vacancy['description'])

    resume_skills = extract_skills(resume_text)

    similarity = skill_similarity(resume_skills, vac_skills)

    resp = f"""
        \nПроцент соответствия навыков: {similarity * 100:.2f}%
        \nИзвлечённые навыки из резюме: {[formatted(s, vac_skills) for s in resume_skills]}
        \nНавыки вакансии: {[formatted(s, resume_skills) for s in vac_skills]}
        """

    bot.reply_to(message, resp)
    
def formatted(skill, skills):
    return f"<b>{skill}</b>" if skill in skills else skill

def build_answer(message, result, resume_text):
    bot.reply_to(message, f'Топ-{RANK} рекомендуемых вакансий:')
    for i, (vacancy_id, score) in enumerate(result):
        vacancy = vacancies[vacancy_id]
        skills = cv_matcher.extract_key_terms(resume_text)
        skill_overlap = cv_matcher.calculate_skill_overlap(vacancy['description'], resume_text)
        confidence = int(score * 100)
        resp = f"""
            {i + 1}. Вакансия #{vacancy_id}: {vacancy['title']}
            ├─ Уверенность подбора: {confidence}%
            ├─ Перекрытие навыков: {int(skill_overlap * 100)}%
            ├─ Основные навыки кандидата: {', '.join(skills[:5]) if skills else 'Не определены'}
            └─ Рекомендация: {'Высокий приоритет' if confidence >= 75 else '✓ Средний приоритет' if confidence >= 50 else 'Низкий приоритет'}
        """
        bot.reply_to(message, f"{resp}")


@bot.edited_message_handler(func=lambda message: True)
def handle_edited_message(message):
    answer(message)

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    answer(message)


bot.polling(none_stop=True, interval=0)
