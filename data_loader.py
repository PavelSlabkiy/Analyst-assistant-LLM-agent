"""
Модуль загрузки данных для аналитического ассистента.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Колонки для удаления при предобработке
COLUMNS_TO_DROP = [
    "type", 
    "offer_type", 
    "description_html", 
    "remote_options", 
    "short_description", 
    "stack_description",
    "short_info",
    "company.logotype",
    "company.url",
    "company.post_to_job_aggregators",
    "recruiter.photo",
    "og.title",
    "og.description",
    "og.image_url",
    "og.image_width",
    "og.image_height",
    "og.site_name",
    "relocation_options",
    "english_level.name",
    "english_level.vacancy_description",
    "one_day_offer_content.version",
    "one_day_offer_content.block_one.header",
    "one_day_offer_content.block_one.last_date",
    "one_day_offer_content.block_one.event_dates",
    "one_day_offer_content.block_one.applications_before",
    "one_day_offer_content.block_two.stack",
    "one_day_offer_content.block_two.header",
    "one_day_offer_content.block_two.short_description",
    "one_day_offer_content.advantages.items",
    "one_day_offer_content.advantages.header",
    "one_day_offer_content_v3.date",
    "one_day_offer_content_v3.teams.items",
    "one_day_offer_content_v3.teams.header",
    "one_day_offer_content_v3.format",
    "one_day_offer_content_v3.schedule.items",
    "one_day_offer_content_v3.schedule.header",
    "one_day_offer_content_v3.block_one.header",
    "one_day_offer_content_v3.block_one.short_description",
    "one_day_offer_content_v3.block_two.stack",
    "one_day_offer_content_v3.block_two.header",
    "one_day_offer_content_v3.block_two.short_description",
    "one_day_offer_content_v3.advantages.items",
    "one_day_offer_content_v3.advantages.header",
    "description",
    "offer_description",
    "analytics_id",
    "office_options",
    "url",
    "company.short_description",
]

# Курсы валют к рублю
CURRENCY_RATES = {
    '₽': 1,
    '$': 80,
    '€': 94,
}


def download_data_from_gdrive(
    url: str = "https://drive.google.com/file/d/1v4aDGZsXmNsFAxQ9D7IoPes-pko7QRzc/view?usp=share_link",
    output_path: str = "data.json"
) -> bool:
    """
    Скачивание данных с Google Drive с помощью gdown.
    
    Аргументы:
        url: URL файла на Google Drive
        output_path: Путь для сохранения скачанного файла
        
    Возвращает:
        True если скачивание успешно, иначе False
    """
    try:
        subprocess.run(
            ["gdown", "--fuzzy", url, "-O", output_path],
            check=True,
            capture_output=True
        )
        print(f"✅ Данные успешно скачаны в {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Не удалось скачать данные: {e}")
        return False
    except FileNotFoundError:
        print("❌ gdown не найден. Установите его командой: pip install gdown")
        return False


def calculate_salary_rub(row: pd.Series, rates: dict = CURRENCY_RATES) -> float:
    """
    Расчёт зарплаты в рублях из различных полей.
    
    Аргументы:
        row: Строка DataFrame с полями зарплаты
        rates: Словарь курсов валют
        
    Возвращает:
        Зарплата в рублях или NaN
    """
    # 1. Определяем значение зарплаты
    frm = row.get('salary_display_from')
    to = row.get('salary_display_to')

    if pd.notna(frm) and pd.notna(to):
        salary = (frm + to) / 2
    elif pd.notna(frm):
        salary = frm
    elif pd.notna(to):
        salary = to
    else:
        # Пытаемся распарсить из salary_description
        text = row.get('salary_description')
        if pd.isna(text):
            return np.nan
        
        match = re.search(r'([\d\s]+)\s*(₽|\$|€)', str(text))
        if not match:
            return np.nan
        
        salary = int(match.group(1).replace(' ', ''))

    # 2. Определяем валюту
    currency = row.get('salary_currency')

    if pd.isna(currency):
        text = row.get('salary_description')
        if pd.isna(text):
            return np.nan
        
        cur_match = re.search(r'(₽|\$|€)', str(text))
        if not cur_match:
            return np.nan
        
        currency = cur_match.group(1)

    # 3. Конвертируем в рубли
    rate = rates.get(currency)
    if rate is None:
        return np.nan

    return salary * rate


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка и предобработка DataFrame.
    
    Аргументы:
        df: Исходный DataFrame
        
    Возвращает:
        Обработанный DataFrame
    """
    # Удаляем колонки, которые существуют в датафрейме
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  ↳ Удалено {len(cols_to_drop)} ненужных колонок")
    
    # Рассчитываем зарплату в рублях
    salary_cols = ['salary_display_from', 'salary_display_to', 'salary_description', 'salary_currency']
    if all(col in df.columns for col in ['salary_display_from', 'salary_display_to']):
        print("  ↳ Расчёт зарплаты в рублях...")
        df['salary'] = df.apply(calculate_salary_rub, axis=1)
        
        # Удаляем исходные колонки зарплаты
        salary_cols_to_drop = [col for col in salary_cols if col in df.columns]
        df = df.drop(columns=salary_cols_to_drop)
        print(f"  ↳ Создана колонка 'salary', удалено {len(salary_cols_to_drop)} исходных колонок зарплаты")
    
    return df


def load_data(data_path: str = "data.json") -> Optional[pd.DataFrame]:
    """
    Загрузка и предобработка JSON-данных в pandas DataFrame.
    
    Аргументы:
        data_path: Путь к JSON-файлу с данными
        
    Возвращает:
        Обработанный DataFrame или None при ошибке загрузки
    """
    path = Path(data_path)
    
    if not path.exists():
        print(f"⚠️ Файл данных не найден: {data_path}")
        print("Попытка скачать с Google Drive...")
        
        if not download_data_from_gdrive(output_path=data_path):
            return None
    
    try:
        print(f"📂 Загрузка данных из {data_path}...")
        data = pd.read_json(path)
        data = pd.json_normalize(data['data'])
        data = data.dropna(axis=1, how="all")
        print(f"  ↳ Загружено {len(data)} записей, {len(data.columns)} колонок")
        
        # Применяем предобработку
        print("🔧 Предобработка данных...")
        data = preprocess_data(data)
        
        print(f"✅ Итоговый датасет: {len(data)} записей, {len(data.columns)} колонок")
        return data
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None


def load_metadata(metadata_path: str = "metadata.json") -> Optional[Dict]:
    """
    Загрузка метаданных из JSON-файла.
    
    Аргументы:
        metadata_path: Путь к JSON-файлу с метаданными
        
    Возвращает:
        Словарь метаданных или None при ошибке загрузки
    """
    path = Path(metadata_path)
    
    if not path.exists():
        print(f"❌ Файл метаданных не найден: {metadata_path}")
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ Загружены метаданные: {len(metadata)} полей")
        return metadata
    except Exception as e:
        print(f"❌ Ошибка загрузки метаданных: {e}")
        return None


if __name__ == "__main__":
    # Тестирование загрузки данных
    df = load_data()
    if df is not None:
        print(f"\nРазмер DataFrame: {df.shape}")
        print(f"Колонки: {list(df.columns)}")
        print(f"\nСтатистика по зарплатам:")
        print(df['salary'].describe())
    
    metadata = load_metadata()
    if metadata is not None:
        print(f"\nПоля метаданных: {list(metadata.keys())}")
