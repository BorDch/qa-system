import os
import gradio as gr
import pandas as pd

from utils.load_data import load_datasets
from utils.generation import generate_answer, load_generation_model
from utils.search import HybridRetrievalModel

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Загрузка данных
print("🚀 Загружаем датасеты...")
know_df, pairs_df, qa_pairs_df = load_datasets(data_path="./data/")

# ───────────────────────────────────────────────────────────────────────────────
# 🔍 Инициализация и подготовка поисковой модели
print("🔧 Подготавливаем модель поиска...")

retrieval_model = HybridRetrievalModel(
    top_k=10,
    use_cross_encoder=True,
    expand_chunks=True,
    verbose=True
)

retrieval_model.prepare(qa_pairs_df, know_df)  # Подготовка по таблице с QA-парами

# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Обработка запроса пользователя
tokenizer, model, device = load_generation_model()

def answer_question(user_question: str) -> str:
    if not user_question.strip():
        return "⚠️ Пожалуйста, введите корректный вопрос."

    # 1. Поиск релевантных чанков в базе знаний
    relevant_chunks = retrieval_model.retrieve(user_question, know_df)

    if not relevant_chunks:
        return "😥 Не удалось найти релевантную информацию. Попробуйте переформулировать вопрос."

    # 2. Сформировать промпт: объединить релевантные чанки и вопрос
    context = "\n".join(relevant_chunks)
    prompt = f"Ответь на вопрос на основе контекста:\n{context}\n\nВопрос: {user_question}\nОтвет:"

    # 3. Генерация ответа
    generated = generate_answer([prompt], tokenizer, model, device)
    return generated[0][0]  # Возврат первой генерации

# ───────────────────────────────────────────────────────────────────────────────
# 💬 Интерфейс Gradio
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Задайте вопрос",
        placeholder="Например: Как получить налоговый вычет?"
    ),
    outputs=gr.Textbox(label="Ответ модели"),
    title="🤖 Система поиска и генерации ответов",
    description="Введите вопрос — система найдет релевантный контекст и сгенерирует ответ на основе найденной информации."
)

# 🚀 Запуск
if __name__ == "__main__":
    iface.launch(share=True)