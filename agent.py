import json

import traceback
from typing import Any, Dict

import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

class LLMAnalystAssistant:
    def __init__(
        self,
        df: pd.DataFrame,
        openrouter_api_key: str,
        metadata: Dict,
        model: str = "kwaipilot/kat-coder-pro:free",
        verbose: bool = True,
    ):
        self.df = df
        self.model = model
        self.verbose = verbose
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.metadata = metadata


    # публичный метод для обращения к классу
    def ask(self, user_prompt: str) -> str:
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

        # если ллм вернула код
        if self._looks_like_code(content):
            code = self._extract_code(content)

            if self.verbose:
                print("\n[CODE TO EXECUTE]")
                print(code)

            result = self.run_with_repair_loop(
                initial_code=code,
                messages=messages,
                max_iterations=3,
            )

            if self.verbose:
                print("\n[RESULT]")
                print(result)

            return str(result)

        # текст
        return content

    # исполнение кода и цикл рефакторинга
    def run_with_repair_loop(
        self,
        initial_code: str,
        messages: list,
        max_iterations: int = 3,
    ) -> Any:
        code = initial_code

        for iteration in range(max_iterations):
            try:
                namespace = {
                    "df": self.df,
                    "pd": pd,
                    "plt": plt,
                    "__builtins__": __builtins__,
                }

                exec(code, namespace, namespace)

                if "result" not in namespace:
                    raise ValueError("Переменная `result` не найдена")

                return namespace["result"]

            except Exception:
                error_text = traceback.format_exc()

                if self.verbose:
                    print(f"\n[ERROR | iteration {iteration + 1}]")
                    print(error_text)

                if iteration == max_iterations - 1:
                    return (
                        "Не удалось выполнить код после нескольких попыток.\n\n"
                        f"{error_text}"
                    )

                # просим модельку починить код
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

    # формируем промпт и метаданные
    def _system_prompt(self, metadata: str) -> str:
        return f"""
Ты — аналитический ассистент по данным.

У тебя есть pandas DataFrame `df` с данными зарплат.

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
   - Построй его
   - result = "График построен"

3. Если вычисления не нужны — верни текст.
"""

    def _build_metadata(self) -> str:
        lines = []
        lines.append("Структура данных:")
        for key in self.metadata.keys():
            lines.append(
                f"field: {key}; "
                f"description: {self.metadata[key]['description']}; "
                f"type: {self.metadata[key]['type']}; "
                f"sample: {self.metadata[key]['sample']}"
            )
        return "\n".join(lines)

    # деетектим код
    @staticmethod
    def _looks_like_code(text: str) -> bool:
        return text.startswith("```")

    @staticmethod
    def _extract_code(text: str) -> str:
        return (
            text.replace("```python", "")
            .replace("```", "")
            .strip()
        )
    

data = pd.read_json('/Users/pavelslabkiy/Desktop/LLM_Assistant/data.json')
data = pd.json_normalize(data['data'])
data = data.dropna(axis=1, how="all")

# инициализируем ассистента
assistant = LLMAnalystAssistant(
    df=data,
    openrouter_api_key="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",
    metadata = metadata,
    verbose=True #режим объяснялкинса шагов, можно оффнуть чтобы не засорять консоль
)