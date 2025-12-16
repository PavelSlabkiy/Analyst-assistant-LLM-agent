"""
LLM Analytics Assistant module.
Generates and executes Python code for data analysis based on user prompts.
"""
import io
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from openai import OpenAI


@dataclass
class AssistantResponse:
    """Response from the assistant containing text and optional attachments."""
    text: str
    image_bytes: Optional[bytes] = None
    xlsx_bytes: Optional[bytes] = None
    xlsx_filename: str = "data_export.xlsx"


@dataclass
class ExecutionResult:
    """Internal result from code execution."""
    raw_result: Any
    result_type: str  # "number", "chart", "table", "text"
    code: str = ""
    image_bytes: Optional[bytes] = None
    xlsx_bytes: Optional[bytes] = None
    xlsx_filename: str = "data_export.xlsx"
    dataframe_info: str = ""  # Info about DataFrame columns/rows
    success: bool = True
    error_message: str = ""


class LLMAnalystAssistant:
    """
    An AI-powered analytics assistant that interprets natural language queries
    and executes Python code for data analysis.
    """
    
    # Model for code generation
    CODE_MODEL = "kwaipilot/kat-coder-pro:free"
    # Model for natural language response formatting
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
        Initialize the analytics assistant.
        
        Args:
            df: The pandas DataFrame to analyze
            openrouter_api_key: API key for OpenRouter
            metadata: Dictionary describing the DataFrame structure
            model: LLM model for code generation
            formatter_model: LLM model for response formatting
            verbose: Whether to print debug information
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
        Process a user's question and return an answer.
        
        Args:
            user_prompt: The user's question in natural language
            
        Returns:
            AssistantResponse with text and optional image/xlsx attachments
        """
        if self.verbose:
            print("\n[USER]", user_prompt)

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
            print("\n[LLM RESPONSE]")
            print(content)

        # If LLM returned code
        if self._looks_like_code(content):
            code = self._extract_code(content)

            if self.verbose:
                print("\n[CODE TO EXECUTE]")
                print(code)

            exec_result = self._run_with_repair_loop(
                initial_code=code,
                messages=messages,
                max_iterations=3,
            )

            # Format the response using the formatter model
            formatted_text = self._format_response(user_prompt, exec_result)

            if self.verbose:
                print("\n[FORMATTED RESULT]")
                print(formatted_text)

            return AssistantResponse(
                text=formatted_text,
                image_bytes=exec_result.image_bytes,
                xlsx_bytes=exec_result.xlsx_bytes,
                xlsx_filename=exec_result.xlsx_filename,
            )

        # Plain text response - still format it nicely
        return AssistantResponse(text=content)

    def _format_response(self, user_question: str, exec_result: ExecutionResult) -> str:
        """
        Format the execution result into a natural language response.
        
        Args:
            user_question: Original user question
            exec_result: Result from code execution
            
        Returns:
            Formatted natural language response
        """
        if not exec_result.success:
            return exec_result.error_message
        
        # Build MINIMAL context for the formatter (avoid token overflow)
        if exec_result.result_type == "number":
            context = f"Вопрос: {user_question}\nРезультат: {exec_result.raw_result}"
        elif exec_result.result_type == "chart":
            context = f"Вопрос: {user_question}"
        elif exec_result.result_type == "table":
            context = f"Вопрос: {user_question}\nТаблица: {exec_result.dataframe_info}"
        else:
            return str(exec_result.raw_result)

        # Create formatting prompt
        format_prompt = self._get_formatter_prompt(exec_result.result_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.formatter_model,
                messages=[
                    {"role": "system", "content": format_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=200,  # Limit output tokens
                stream=False,
            )
            
            formatted = response.choices[0].message.content.strip()
            
            if self.verbose:
                print("\n[FORMATTER RESPONSE]")
                print(formatted)
            
            return formatted
            
        except Exception as e:
            if self.verbose:
                print(f"\n[FORMATTER ERROR] {e}")
            # Fallback to raw result if formatting fails
            return str(exec_result.raw_result)

    def _get_formatter_prompt(self, result_type: str) -> str:
        """Get the system prompt for the formatter based on result type."""
        
        if result_type == "number":
            return """Сформулируй ответ на русском (1-2 предложения).
Форматируй числа: пробелы в тысячах (150 000), проценты (0.15→15%), зарплаты в ₽.
Пример: "Средняя зарплата составляет 185 000 ₽." """

        elif result_type == "chart":
            return """Опиши построенный график на русском (1-2 предложения).
Укажи тип графика и что визуализировано. Добавь emoji 📊 или 📈.
Пример: "📊 Построена диаграмма зарплат по городам." """

        elif result_type == "table":
            return """Опиши выгрузку на русском (1-2 предложения).
Укажи кол-во строк и основные колонки. Упомяни Excel. Добавь emoji 📋.
Пример: "📋 Выгружено 10 вакансий (позиция, зарплата, город). Excel прикреплён." """

        return "Ответь кратко на русском."

    def _run_with_repair_loop(
        self,
        initial_code: str,
        messages: list,
        max_iterations: int = 3,
    ) -> ExecutionResult:
        """
        Execute code with automatic error repair loop.
        
        Args:
            initial_code: The initial Python code to execute
            messages: Conversation history for context
            max_iterations: Maximum repair attempts
            
        Returns:
            ExecutionResult with results
        """
        code = initial_code

        for iteration in range(max_iterations):
            try:
                # Create isolated namespace with necessary imports
                namespace = {
                    "df": self.df,
                    "pd": pd,
                    "plt": plt,
                    "io": io,
                    "__builtins__": __builtins__,
                }

                # Close any existing figures
                plt.close('all')
                
                exec(code, namespace, namespace)

                if "result" not in namespace:
                    raise ValueError("Variable `result` not found in code output")

                result = namespace["result"]
                
                # Determine result type and process accordingly
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
                    print(f"\n[ERROR | iteration {iteration + 1}]")
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

                # Ask the model to fix the code
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
                    print("\n[REPAIRED CODE]")
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
        Create an Excel file from a DataFrame.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Tuple of (xlsx bytes, filename)
        """
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Данные')
        buf.seek(0)
        
        # Generate filename based on columns
        cols = '_'.join(df.columns[:2].tolist())[:30] if len(df.columns) > 0 else 'data'
        cols = ''.join(c if c.isalnum() or c == '_' else '_' for c in cols)
        filename = f"{cols}_export.xlsx"
        
        return buf.getvalue(), filename

    def _capture_plot(self) -> Optional[bytes]:
        """
        Capture the current matplotlib figure as PNG bytes.
        
        Returns:
            PNG image bytes or None if no figure exists
        """
        fig = plt.gcf()
        if fig.get_axes():  # Check if figure has any axes (i.e., a plot was created)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close('all')
            return buf.getvalue()
        return None

    def _system_prompt(self, metadata: str) -> str:
        """Generate the system prompt for the LLM."""
        return f"""
Ты — аналитический ассистент по данным о вакансиях и зарплатах.

У тебя есть pandas DataFrame `df` с данными.

{metadata}

ПРАВИЛА (СТРОГО):

1. Если вопрос требует ОДНОГО числа (среднее, медиана, количество, сумма):
   - Верни Python-код, где result = число или строка с числом
   - Пример: result = df['salary_display_from'].mean()

2. Если нужно построить ГРАФИК (слова: график, диаграмма, визуализация, покажи на графике):
   - Используй matplotlib (plt)
   - Настрой шрифты: plt.rcParams['font.family'] = 'DejaVu Sans'
   - Добавь заголовок и подписи осей
   - Используй plt.figure(figsize=(10, 6))
   - result = "График построен"

3. Если нужна ТАБЛИЦА, ВЫГРУЗКА, ЭКСПОРТ, СПИСОК, ДИНАМИКА, ТОП (без графика):
   - Слова-триггеры: таблица, выгрузи, экспорт, список, покажи данные, топ-N, динамика, по месяцам, по дням
   - result должен быть DataFrame (pd.DataFrame)
   - Пример: result = df[['position', 'salary_display_from']].head(10)
   - Пример: result = df.groupby('city').agg({{'salary_display_from': 'mean'}}).reset_index()

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
        """Build metadata description string from metadata dictionary."""
        lines = ["Структура данных:"]
        for key, value in self.metadata.items():
            lines.append(
                f"- `{key}`: {value.get('description', 'N/A')} "
                f"(тип: {value.get('type', 'N/A')}, пример: {value.get('sample', 'N/A')})"
            )
        return "\n".join(lines)

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """Check if the response looks like code."""
        return "```" in text

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from markdown code blocks."""
        # Handle ```python ... ``` format
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code_part = parts[1].split("```")[0]
                return code_part.strip()
        
        # Handle ``` ... ``` format
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].strip()
        
        return text.strip()
