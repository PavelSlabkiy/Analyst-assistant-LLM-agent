"""
Модуль LLM-ассистента для аналитики.
Генерирует и выполняет Python-код для анализа данных на основе запросов пользователя.
"""
import io
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд для серверного окружения
import matplotlib.pyplot as plt
from openai import OpenAI


@dataclass
class AssistantResponse:
    """Ответ ассистента с текстом и опциональными вложениями."""
    text: str
    image_bytes: Optional[bytes] = None
    xlsx_bytes: Optional[bytes] = None
    xlsx_filename: str = "data_export.xlsx"


@dataclass
class ExecutionResult:
    """Внутренний результат выполнения кода."""
    raw_result: Any
    result_type: str  # "number", "chart", "table", "text"
    code: str = ""
    image_bytes: Optional[bytes] = None
    xlsx_bytes: Optional[bytes] = None
    xlsx_filename: str = "data_export.xlsx"
    dataframe_info: str = ""  # Информация о колонках/строках DataFrame
    success: bool = True
    error_message: str = ""


class LLMAnalystAssistant:
    """
    ИИ-ассистент для аналитики, который интерпретирует запросы на естественном языке
    и выполняет Python-код для анализа данных.
    """
    
    # Модель для генерации кода
    CODE_MODEL = "kwaipilot/kat-coder-pro:free"
    # Модель для форматирования ответов на естественном языке
    FORMATTER_MODEL = "google/gemma-3-4b-it:free"
    
    def __init__(
        self,
        df: pd.DataFrame,
        openrouter_api_key: str,
        metadata: Dict,
        model: str = "kwaipilot/kat-coder-pro:free",
        formatter_model: str = "google/gemma-3-4b-it:free",
        verbose: bool = False,
    ):
        """
        Инициализация аналитического ассистента.
        
        Аргументы:
            df: pandas DataFrame для анализа
            openrouter_api_key: API-ключ для OpenRouter
            metadata: Словарь с описанием структуры DataFrame
            model: Модель LLM для генерации кода
            formatter_model: Модель LLM для форматирования ответов
            verbose: Выводить ли отладочную информацию
        """
        self.df = df
        self.model = model
        self.formatter_model = formatter_model
        self.verbose = verbose
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.metadata = metadata

    def ask(self, user_prompt: str) -> AssistantResponse:
        """
        Обработка вопроса пользователя и возврат ответа.
        
        Аргументы:
            user_prompt: Вопрос пользователя на естественном языке
            
        Возвращает:
            AssistantResponse с текстом и опциональными вложениями (изображение/xlsx)
        """
        if self.verbose:
            print("\n[ПОЛЬЗОВАТЕЛЬ]", user_prompt)

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(self._build_metadata()),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stream=False,
        )

        content = response.choices[0].message.content.strip()

        if self.verbose:
            print("\n[ОТВЕТ LLM]")
            print(content)

        # Если LLM вернула код
        if self._looks_like_code(content):
            code = self._extract_code(content)

            if self.verbose:
                print("\n[КОД ДЛЯ ВЫПОЛНЕНИЯ]")
                print(code)

            exec_result = self._run_with_repair_loop(
                initial_code=code,
                messages=messages,
                max_iterations=3,
            )

            # Форматируем ответ с помощью модели-форматтера
            formatted_text = self._format_response(user_prompt, exec_result)

            if self.verbose:
                print("\n[ОТФОРМАТИРОВАННЫЙ РЕЗУЛЬТАТ]")
                print(formatted_text)

            return AssistantResponse(
                text=formatted_text,
                image_bytes=exec_result.image_bytes,
                xlsx_bytes=exec_result.xlsx_bytes,
                xlsx_filename=exec_result.xlsx_filename,
            )

        # Текстовый ответ без кода
        return AssistantResponse(text=content)

    def _format_response(self, user_question: str, exec_result: ExecutionResult) -> str:
        """
        Форматирование результата выполнения в ответ на естественном языке.
        
        Аргументы:
            user_question: Исходный вопрос пользователя
            exec_result: Результат выполнения кода
            
        Возвращает:
            Отформатированный ответ на естественном языке
        """
        if not exec_result.success:
            return exec_result.error_message
        
        # Собираем МИНИМАЛЬНЫЙ контекст для форматтера (избегаем переполнения токенов)
        if exec_result.result_type == "number":
            context = f"Вопрос: {user_question}\nРезультат: {exec_result.raw_result}"
        elif exec_result.result_type == "chart":
            context = f"Вопрос: {user_question}\nКод: {exec_result.code}"
        elif exec_result.result_type == "table":
            context = f"Вопрос: {user_question}\nСтрок: {exec_result.dataframe_info.split(',')[0] if exec_result.dataframe_info else 'N/A'}\nКод: {exec_result.code}"
        else:
            return str(exec_result.raw_result)

        # Получаем промпт для форматирования
        format_prompt = self._get_formatter_prompt(exec_result.result_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.formatter_model,
                messages=[
                    {"role": "system", "content": format_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=200,  # Ограничиваем количество токенов на выходе
                stream=False,
            )
            
            formatted = response.choices[0].message.content.strip()
            
            if self.verbose:
                print("\n[ОТВЕТ ФОРМАТТЕРА]")
                print(formatted)
            
            return formatted
            
        except Exception as e:
            if self.verbose:
                print(f"\n[ОШИБКА ФОРМАТТЕРА] {e}")
            # Возвращаем сырой результат, если форматирование не удалось
            return str(exec_result.raw_result)

    def _get_formatter_prompt(self, result_type: str) -> str:
        """Получение системного промпта для форматтера в зависимости от типа результата."""
        
        if result_type == "number":
            return """Сформулируй ответ на русском (1-2 предложения).
Форматируй числа: пробелы в тысячах (150 000), проценты (0.15→15%), зарплаты в ₽.
Пример: "Средняя зарплата составляет 185 000 ₽." """

        elif result_type == "chart":
            return """Опиши график на русском. 
Не возвращай данные графика, только опиши какой график построен и какие поля из кода использовались.
Пример: "📊 Построена диаграмма средних зарплат (salary) по городам (city)." """

        elif result_type == "table":
            return """Опиши выгрузку на русском. Не возвращай данные таблицы, только опиши что выгружено, сколько строк, какие поля из кода использовались.
Пример: "📋 Выгружено 10 записей с полями position, salary, city. Excel прикреплён." """

        return "Ответь кратко на русском."

    def _run_with_repair_loop(
        self,
        initial_code: str,
        messages: list,
        max_iterations: int = 3,
    ) -> ExecutionResult:
        """
        Выполнение кода с автоматическим циклом исправления ошибок.
        
        Аргументы:
            initial_code: Исходный Python-код для выполнения
            messages: История диалога для контекста
            max_iterations: Максимальное количество попыток исправления
            
        Возвращает:
            ExecutionResult с результатами выполнения
        """
        code = initial_code

        for iteration in range(max_iterations):
            try:
                # Создаём изолированное пространство имён с необходимыми импортами
                namespace = {
                    "df": self.df,
                    "pd": pd,
                    "plt": plt,
                    "io": io,
                    "__builtins__": __builtins__,
                }

                # Закрываем все существующие графики
                plt.close('all')
                
                exec(code, namespace, namespace)

                if "result" not in namespace:
                    raise ValueError("Переменная `result` не найдена в результате выполнения кода")

                result = namespace["result"]
                
                # Определяем тип результата и обрабатываем соответственно
                image_bytes = self._capture_plot()
                xlsx_bytes = None
                xlsx_filename = "data_export.xlsx"
                dataframe_info = ""
                
                if isinstance(result, pd.DataFrame):
                    result_type = "table"
                    xlsx_bytes, xlsx_filename = self._create_xlsx(result)
                    dataframe_info = f"{len(result)} строк, колонки: {', '.join(result.columns.tolist()[:10])}"
                elif image_bytes:
                    result_type = "chart"
                elif isinstance(result, (int, float)) or (isinstance(result, str) and any(c.isdigit() for c in result)):
                    result_type = "number"
                else:
                    result_type = "text"

                return ExecutionResult(
                    raw_result=result,
                    result_type=result_type,
                    code=code,
                    image_bytes=image_bytes,
                    xlsx_bytes=xlsx_bytes,
                    xlsx_filename=xlsx_filename,
                    dataframe_info=dataframe_info,
                    success=True,
                )

            except Exception:
                error_text = traceback.format_exc()

                if self.verbose:
                    print(f"\n[ОШИБКА | попытка {iteration + 1}]")
                    print(error_text)

                if iteration == max_iterations - 1:
                    return ExecutionResult(
                        raw_result=None,
                        result_type="error",
                        success=False,
                        error_message=(
                            "🤔 К сожалению, бот пока не знает ответа на этот вопрос.\n\n"
                            "Попробуйте переформулировать запрос или задать другой вопрос."
                        ),
                    )

                # Просим модель исправить код
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Вот код, который ты написал:\n```python\n{code}\n```",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "В этом коде произошла ошибка:\n"
                            f"```text\n{error_text}\n```\n\n"
                            "Исправь код. Верни ТОЛЬКО исправленный Python-код. "
                            "Используй df и сохрани результат в переменную result."
                        ),
                    }
                )

                repair_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    stream=False,
                )

                code = self._extract_code(
                    repair_response.choices[0].message.content
                )

                if self.verbose:
                    print("\n[ИСПРАВЛЕННЫЙ КОД]")
                    print(code)

        return ExecutionResult(
            raw_result=None,
            result_type="error",
            success=False,
            error_message=(
                "🤔 К сожалению, бот пока не знает ответа на этот вопрос.\n\n"
                "Попробуйте переформулировать запрос или задать другой вопрос."
            ),
        )

    def _create_xlsx(self, df: pd.DataFrame) -> tuple[bytes, str]:
        """
        Создание Excel-файла из DataFrame.
        
        Аргументы:
            df: DataFrame для экспорта
            
        Возвращает:
            Кортеж (байты xlsx-файла, имя файла)
        """
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Данные')
        buf.seek(0)
        
        # Генерируем имя файла на основе колонок
        cols = '_'.join(df.columns[:2].tolist())[:30] if len(df.columns) > 0 else 'data'
        cols = ''.join(c if c.isalnum() or c == '_' else '_' for c in cols)
        filename = f"{cols}_export.xlsx"
        
        return buf.getvalue(), filename

    def _capture_plot(self) -> Optional[bytes]:
        """
        Захват текущего графика matplotlib в виде PNG-байтов.
        
        Возвращает:
            PNG-изображение в байтах или None, если график не был создан
        """
        fig = plt.gcf()
        if fig.get_axes():  # Проверяем, есть ли у фигуры оси (т.е. был ли создан график)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close('all')
            return buf.getvalue()
        return None

    def _system_prompt(self, metadata: str) -> str:
        """Генерация системного промпта для LLM."""
        return f"""
Ты — аналитический ассистент по данным о вакансиях и зарплатах.

У тебя есть pandas DataFrame `df` с данными.

{metadata}

ПРАВИЛА (СТРОГО):

1. Если вопрос требует ОДНОГО числа (среднее, медиана, количество, сумма):
   - Верни Python-код, где result = число или строка с числом
   - Пример: result = df['salary'].mean()

2. Если нужно построить ГРАФИК (слова: график, диаграмма, визуализация, покажи на графике):
   - Используй matplotlib (plt)
   - Настрой шрифты: plt.rcParams['font.family'] = 'DejaVu Sans'
   - Добавь заголовок и подписи осей
   - Используй plt.figure(figsize=(10, 6))
   - result = "График построен"

3. Если нужна ТАБЛИЦА, ВЫГРУЗКА, ЭКСПОРТ, СПИСОК, ДИНАМИКА, ТОП (без графика):
   - Слова-триггеры: таблица, выгрузи, экспорт, список, покажи данные, топ-N, динамика, по месяцам, по дням
   - result должен быть DataFrame (pd.DataFrame)
   - Пример: result = df[['position', 'salary']].head(10)
   - Пример: result = df.groupby('city').agg({{'salary': 'mean'}}).reset_index()

4. Если вычисления не нужны — верни текст без кода.

ФОРМАТ КОДА:
```python
<твой код>```

ВАЖНО:
- Всегда сохраняй результат в переменную `result`
- Для таблиц: result = DataFrame
- Для чисел: result = число или f-строка
- Для графиков: result = "График построен"
"""

    def _build_metadata(self) -> str:
        """Формирование строки с описанием метаданных из словаря."""
        lines = ["Структура данных:"]
        for key, value in self.metadata.items():
            lines.append(
                f"- `{key}`: {value.get('description', 'N/A')} "
                f"(тип: {value.get('type', 'N/A')}, пример: {value.get('sample', 'N/A')})"
            )
        return "\n".join(lines)

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Проверка, похож ли ответ на код."""
        return "```" in text

    @staticmethod
    def _extract_code(text: str) -> str:
        """Извлечение Python-кода из markdown-блоков."""
        # Обработка формата ```python ... ```
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        
        # Обработка формата ``` ... ```
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].strip()
        
        return text.strip()
