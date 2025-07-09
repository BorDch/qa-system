"""
search.py

Модуль гибридного поиска для задачи генерации ответов на основе внешнего контекста.
Объединяет возможности BM25, bi-encoder и cross-encoder моделей для извлечения наиболее релевантных текстов.

Зависимости:
- sentence-transformers
- rank_bm25
- sklearn
- pandas, numpy, torch
"""
import pandas as pd
from tqdm import tqdm
import re
import torch
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


class HybridRetrievalModel:
    """
    Класс гибридного поиска, объединяющий BM25, bi-encoder и (опционально) cross-encoder
    для извлечения наиболее релевантных текстов по заданному вопросу.

    Атрибуты:
    ----------
    top_k : int
        Количество кандидатов для rerank-а.
    model : SentenceTransformer
        Bi-encoder модель для получения эмбеддингов.
    cross_encoder : CrossEncoder
        Модель для переоценки релевантности (если включена).
    use_cross_encoder : bool
        Использовать ли cross-encoder в pipeline.
    expand_chunks : bool
        Расширять ли выбранный chunk соседними (из chunk_df).
    chunk_df : pd.DataFrame
        DataFrame с chunk'ами и колонками `document_id`, `chunk`.
    verbose : bool
        Печатать ли логи процесса.
    """

    def __init__(
        self,
        top_k=10,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cross_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder=True,
        expand_chunks=False,
        chunk_df=None,
        verbose=False
    ):
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cross_encoder = CrossEncoder(cross_model_name, device=self.device) if use_cross_encoder else None
        self.use_cross_encoder = use_cross_encoder
        self.expand_chunks = expand_chunks
        self.chunk_df = chunk_df
        self.verbose = verbose

        self.bm25 = None
        self.corpus_embeddings = None
        self.doc_ids = None
        self.train_texts = None
        self.qa_df = None

    def preprocess(self, text):
        """
        Предобработка текста: нижний регистр, удаление пунктуации, токенизация.

        Parameters
        ----------
        text : str
            Входной текст.

        Returns
        -------
        List[str]
            Токены текста.
        """
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def prepare(self, qa_df, know_df):
        """
        Подготовка корпуса: извлечение текстов, токенизация, получение эмбеддингов, инициализация BM25.

        Parameters
        ----------
        qa_df : pd.DataFrame
            DataFrame с колонками `question`, `answer`, `document_id`.
        """

        # Токенизация
        def tokenize(text):
            return str(text).lower().split()

        # Индексация всех chunks из know_df
        know_df['tokens'] = know_df['chunk'].apply(tokenize)
        bm25_chunks = BM25Okapi(know_df['tokens'].tolist())

        # Привязка question → document_id
        def map_question_to_doc_id(question, k=1):
            tokens = tokenize(question)
            scores = bm25_chunks.get_scores(tokens)
            best_idx = np.argmax(scores)
            return know_df.iloc[best_idx]['document_id']

        # Применим к qa_pairs_df
        tqdm.pandas(desc="🔗 Matching to document_id")
        qa_df['document_id'] = qa_df['question'].progress_apply(map_question_to_doc_id)

        qa_df = qa_df.dropna(subset=["question", "answer", "document_id"]).reset_index(drop=True)
        self.qa_df = qa_df.copy()

        self.train_texts = (qa_df["question"] + " " + qa_df["answer"]).astype(str).tolist()
        self.tokenized_corpus = [self.preprocess(text) for text in self.train_texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.corpus_embeddings = self.model.encode(
            self.train_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.doc_ids = qa_df["document_id"].tolist()

        if self.verbose:
            print(f"✅ Корпус подготовлен: {len(self.train_texts)} текстов")

    def expand_chunk_by_neighbors(self, idx, window=1):
        """
        Расширение текста соседними chunk'ами (в пределах документа).

        Parameters
        ----------
        idx : int
            Индекс центрального chunk'а.
        window : int, optional
            Число соседей слева и справа (по умолчанию 1).

        Returns
        -------
        str
            Расширенный текст.
        """
        if self.chunk_df is None:
            return self.train_texts[idx]

        doc_id = self.doc_ids[idx]
        doc_chunks = self.chunk_df[self.chunk_df["document_id"] == doc_id].reset_index(drop=True)

        try:
            center_idx = idx % len(doc_chunks)
        except ZeroDivisionError:
            return self.train_texts[idx]

        start = max(0, center_idx - window)
        end = min(len(doc_chunks), center_idx + window + 1)
        return " ".join(doc_chunks.iloc[start:end]["chunk"].tolist())

    def search(self, query, return_all=False):
        """
        Поиск релевантных текстов по запросу с использованием BM25 + bi-encoder + (опционально) cross-encoder.

        Parameters
        ----------
        query : str
            Входной вопрос.
        return_all : bool, optional
            Вернуть ли топ-N результатов с метриками (по умолчанию False).

        Returns
        -------
        Union[dict, pd.DataFrame]
            Если return_all=False: словарь с лучшим результатом.
            Если return_all=True: DataFrame с топ-N кандидатами и их метриками.
        """
        if self.bm25 is None:
            raise ValueError("Сначала вызовите метод `.prepare()` для подготовки корпуса.")

        query_tokens = self.preprocess(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[-self.top_k:][::-1]

        query_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        candidate_embs = self.corpus_embeddings[top_indices]
        sims = cosine_similarity(query_emb, candidate_embs)[0]

        candidates = [self.train_texts[i] for i in top_indices]
        doc_ids_top = [self.doc_ids[i] for i in top_indices]

        if self.use_cross_encoder:
            cross_scores = self.cross_encoder.predict([(query, cand) for cand in candidates])
            rerank_scores = cross_scores
        else:
            rerank_scores = sims

        best_idx_rel = int(np.argmax(rerank_scores))
        best_idx = top_indices[best_idx_rel]

        final_chunk = self.expand_chunk_by_neighbors(best_idx) if self.expand_chunks else self.train_texts[best_idx]

        if return_all:
            results_df = pd.DataFrame({
                "doc_id": doc_ids_top,
                "text": candidates,
                "bm25_score": [bm25_scores[i] for i in top_indices],
                "cosine_sim": sims,
                "rerank_score": rerank_scores,
            })
            return results_df.sort_values(by="rerank_score", ascending=False).reset_index(drop=True)

        return {
            "best_doc_id": self.doc_ids[best_idx],
            "best_chunk": final_chunk,
            "bm25_rank": best_idx,
            "bm25_score": bm25_scores[best_idx],
        }

    def retrieve(self, query, know_df=None):
        """
        Метод-обертка для получения релевантного текста (chunk) по запросу пользователя.

        Parameters
        ----------
        query : str
            Вопрос пользователя.
        know_df : pd.DataFrame, optional
            Необязательный DataFrame с chunk'ами. Используется, если модель была инициализирована без chunk_df.

        Returns
        -------
        List[str]
            Список с одним релевантным текстом (chunk).
        """
        result = self.search(query, return_all=False)
        return [result["best_chunk"]]