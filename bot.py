"""
Telegram Bot for the Analytics Assistant.
Uses Aiogram 3.x for async Telegram bot functionality.
"""
import asyncio
import logging
import sys
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile
from aiogram.client.default import DefaultBotProperties

from config import config
from agent import LLMAnalystAssistant
from data_loader import load_data, load_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Initialize router
router = Router()

# Global assistant instance (initialized in main)
assistant: Optional[LLMAnalystAssistant] = None


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command."""
    welcome_text = """
👋 <b>Привет! Я аналитический ассистент по данным о вакансиях.</b>

Я могу помочь тебе с анализом данных о вакансиях и зарплатах. 

<b>Что я умею:</b>
📊 Считать статистику (средние, медианы, распределения)
📈 Строить графики и диаграммы  
📋 Выгружать данные в Excel-таблицы
🔍 Фильтровать и группировать данные
💰 Анализировать зарплаты по городам, специализациям и уровням

<b>Примеры вопросов:</b>
• Какая средняя зарплата для Data Engineer?
• Построй график распределения зарплат по городам
• Выгрузи топ-20 вакансий с самой высокой зарплатой
• Сколько вакансий в Москве?
• Покажи динамику вакансий по месяцам в виде таблицы

Просто напиши свой вопрос! 💬
"""
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command."""
    help_text = """
📚 <b>Справка по использованию бота</b>

<b>Команды:</b>
/start - Начать работу с ботом
/help - Показать эту справку
/info - Информация о загруженных данных

<b>Как задавать вопросы:</b>
Пишите вопросы на русском языке. Бот понимает контекст и может:
- Выполнять вычисления → получите число
- Строить графики → получите изображение
- Выгружать таблицы → получите Excel-файл

<b>Примеры для разных типов ответов:</b>

📊 <b>Число:</b>
• "Какая средняя зарплата?"
• "Сколько всего вакансий?"

📈 <b>График:</b>
• "Построй график зарплат по городам"
• "Покажи распределение на диаграмме"

📋 <b>Таблица (Excel):</b>
• "Выгрузи топ-10 вакансий"
• "Покажи таблицу зарплат по специализациям"
• "Экспортируй данные по Python-вакансиям"
• "Покажи динамику по месяцам"

<b>Советы:</b>
• Для графика используйте слова: график, диаграмма, визуализация
• Для таблицы: выгрузи, экспорт, таблица, список, топ-N
"""
    await message.answer(help_text, parse_mode=ParseMode.HTML)


@router.message(Command("info"))
async def cmd_info(message: Message) -> None:
    """Handle /info command - show data info."""
    if assistant is None:
        await message.answer("❌ Данные не загружены")
        return
    
    df = assistant.df
    info_text = f"""
📊 <b>Информация о данных</b>

📁 <b>Записей:</b> {len(df):,}
📋 <b>Полей:</b> {len(df.columns)}

<b>Основные поля:</b>
• position - название позиции
• specialization - специализация
• position_level - уровень (Junior/Middle/Senior)
• salary_display_from/to - диапазон зарплаты
• city - город
• country - страна
• stack - технологический стек
"""
    await message.answer(info_text, parse_mode=ParseMode.HTML)


@router.message(F.text)
async def handle_question(message: Message) -> None:
    """Handle user questions."""
    if assistant is None:
        await message.answer("❌ Ассистент не инициализирован. Попробуйте позже.")
        return
    
    user_question = message.text.strip()
    
    if not user_question:
        await message.answer("❓ Пожалуйста, задайте вопрос.")
        return
    
    # Send "typing" status
    await message.answer("🔄 Обрабатываю ваш запрос...")
    
    try:
        # Run the assistant in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            assistant.ask,
            user_question
        )
        
        # Send the text response
        if response.text:
            # Escape special characters for HTML
            safe_text = (
                response.text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            
            # Truncate very long responses
            if len(safe_text) > 4000:
                safe_text = safe_text[:4000] + "\n\n... (ответ обрезан)"
            
            await message.answer(f"📋 <b>Результат:</b>\n\n<code>{safe_text}</code>", parse_mode=ParseMode.HTML)
        
        # Send the image if one was generated
        if response.image_bytes:
            photo = BufferedInputFile(response.image_bytes, filename="chart.png")
            await message.answer_photo(photo, caption="📈 График по вашему запросу")
        
        # Send the Excel file if one was generated
        if response.xlsx_bytes:
            document = BufferedInputFile(response.xlsx_bytes, filename=response.xlsx_filename)
            await message.answer_document(document, caption="📋 Данные в формате Excel")
            
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        await message.answer(
            f"❌ Произошла ошибка при обработке запроса.\n\n"
            f"<code>{str(e)[:500]}</code>",
            parse_mode=ParseMode.HTML
        )


async def main() -> None:
    """Main function to run the bot."""
    global assistant
    
    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Load data and metadata
    logger.info("Loading data...")
    df = load_data(config.DATA_PATH)
    if df is None:
        logger.error("Failed to load data. Make sure data.json exists or can be downloaded.")
        sys.exit(1)
    
    metadata = load_metadata(config.METADATA_PATH)
    if metadata is None:
        logger.error("Failed to load metadata. Make sure metadata.json exists.")
        sys.exit(1)
    
    # Initialize the assistant
    logger.info("Initializing LLM assistant...")
    assistant = LLMAnalystAssistant(
        df=df,
        openrouter_api_key=config.OPENROUTER_API_KEY,
        metadata=metadata,
        model=config.LLM_MODEL,
        verbose=True,
    )
    
    # Initialize bot and dispatcher
    logger.info("Starting Telegram bot...")
    bot = Bot(
        token=config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(router)
    
    # Start polling
    logger.info("Bot is running! Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
