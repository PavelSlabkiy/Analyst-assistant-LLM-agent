"""
LLM Analytics Assistant module.
Generates and executes Python code for data analysis based on user prompts.
"""
import io
import traceback
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from openai import OpenAI


class LLMAnalystAssistant:
    """
    An AI-powered analytics assistant that interprets natural language queries
    and executes Python code for data analysis.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        openrouter_api_key: str,
        metadata: Dict,
        model: str = "kwaipilot/kat-coder-pro:free",
        verbose: bool = False,
    ):
        """
        Initialize the analytics assistant.
        
        Args:
            df: The pandas DataFrame to analyze
            openrouter_api_key: API key for OpenRouter
            metadata: Dictionary describing the DataFrame structure
            model: LLM model identifier
            verbose: Whether to print debug information
        """
        self.df = df
        self.model = model
        self.verbose = verbose
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.metadata = metadata

    def ask(self, user_prompt: str) -> Tuple[str, Optional[bytes]]:
        """
        Process a user's question and return an answer.
        
        Args:
            user_prompt: The user's question in natural language
            
        Returns:
            Tuple of (text_response, image_bytes or None)
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

            result, image_bytes = self._run_with_repair_loop(
                initial_code=code,
                messages=messages,
                max_iterations=3,
            )

            if self.verbose:
                print("\n[RESULT]")
                print(result)

            return str(result), image_bytes

        # Plain text response
        return content, None

    def _run_with_repair_loop(
        self,
        initial_code: str,
        messages: list,
        max_iterations: int = 3,
    ) -> Tuple[Any, Optional[bytes]]:
        """
        Execute code with automatic error repair loop.
        
        Args:
            initial_code: The initial Python code to execute
            messages: Conversation history for context
            max_iterations: Maximum repair attempts
            
        Returns:
            Tuple of (result, image_bytes or None)
        """
        code = initial_code

        for iteration in range(max_iterations):
            try:
                # Create isolated namespace with necessary imports
                namespace = {
                    "df": self.df,
                    "pd": pd,
                    "plt": plt,
                    "__builtins__": __builtins__,
                }

                # Close any existing figures
                plt.close('all')
                
                exec(code, namespace, namespace)

                if "result" not in namespace:
                    raise ValueError("Variable `result` not found in code output")

                result = namespace["result"]
                
                # Check if a plot was created
                image_bytes = self._capture_plot()

                return result, image_bytes

            except Exception:
                error_text = traceback.format_exc()

                if self.verbose:
                    print(f"\n[ERROR | iteration {iteration + 1}]")
                    print(error_text)

                if iteration == max_iterations - 1:
                    return (
                        f"❌ Не удалось выполнить код после {max_iterations} попыток.\n\n"
                        f"Ошибка: {error_text}"
                    ), None

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

        return "Unexpected error in repair loop", None

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

1. Если вопрос требует вычислений или агрегаций:
   - Верни ТОЛЬКО Python-код
   - Без объяснений
   - Используй df
   - Сохрани результат в переменную `result`
   - Формат для кода:
```python
<твой код>```

2. Если нужно построить график:
   - Используй matplotlib (plt)
   - Настрой русские шрифты: plt.rcParams['font.family'] = 'DejaVu Sans'
   - Добавь заголовок и подписи осей
   - Используй plt.figure(figsize=(10, 6)) для читаемости
   - result = "График построен"

3. Если вычисления не нужны — верни текст.

4. Для числовых результатов — форматируй красиво (разделители тысяч, округление).
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
