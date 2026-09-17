# genius-rag

Agentic RAG над аннотациями [Genius](https://genius.com) API: гибридный поиск,
LangGraph-агент с инструментами и честная оценка качества (аннотации как ground truth).

## 1. Что делает

Источник — Genius API: артисты → треки → referents (фрагмент строки) + annotations
(объяснение сообщества) + метаданные (альбом, фиты, продюсеры, сэмплы/каверы).
Полные тексты не хранятся.

- `POST /ask` — вопрос → LangGraph-агент: роутер → инструменты → grounded ответ со ссылками на аннотации
  - «что значит строчка "…" у Oxxxymiron?» → vector_search по аннотациям
  - «сколько треков у Скриптонита с фитами Truwer?» → sql_query по Postgres
  - «кого сэмплирует трек X и что об этом пишут?» → song_graph + vector_search (multi-hop)
- `POST /annotate` — строка без аннотации → черновик объяснения по похожим аннотациям (few-shot из retrieval)
- eval: retrieval (recall@k, MRR, nDCG), A/B чанкинга, RAGAS faithfulness/relevancy,
  LLM-judge «черновик vs реальная аннотация» → таблица метрик в README
- Telegram-бот (aiogram 3) поверх `/ask`

## 2. Не делает

- полные тексты песен (копирайт), UI, авторизация
- весь Genius — 10–20 артистов (ru + en), ~2–5k аннотаций

## 3. Архитектура

```
Genius API ─> ingest (httpx, rate-limit) ─> data/raw/*.jsonl ─> Pandas EDA
                                              ├─> Postgres: artists / songs / song_relationships / annotations
                                              └─> contextual chunks ─> e5 dense + BM25 sparse ─> Qdrant (hybrid)

LangGraph agent:
  route ─> [vector_search | sql_query | song_graph] ─> rerank ─> grade ─> answer(+citations)
                                  ▲                                 │ low coverage
                                  └───────────── retry ─────────────┘

FastAPI /ask (SSE), /annotate   ·   aiogram bot   ·   eval/   ·   Langfuse (опц.)
```

## 4. Стек

| Слой | Выбор | Почему |
|---|---|---|
| Язык | Python 3.12+, `uv` | стандарт |
| Оркестрация | LangChain + LangGraph, agent с tool-calling | маршрут зависит от вопроса |
| LLM | OpenRouter via `langchain-openai`; GigaChat опц. | абстракция провайдера |
| Embeddings | `intfloat/multilingual-e5-small` + contextual prefix | ru+en, CPU |
| Sparse | BM25 через `fastembed` | hybrid без отдельного движка |
| Vector store | Qdrant, native hybrid + payload-фильтры | |
| Метаданные | PostgreSQL 16 — связи треков, text-to-SQL tool | |
| Reranker | `BAAI/bge-reranker-v2-m3` | cross-encoder точность |
| Данные | Genius API → JSONL, Pandas для EDA | |
| API | FastAPI + SSE | REST/JSON |
| Чат | aiogram 3 | |
| Eval | pytest + свой harness + RAGAS + LLM-judge | ground truth = аннотации |
| Observability | Langfuse (опц., docker) | трейсы агента |
| Инфра | Docker Compose (app, qdrant, postgres), Makefile, `.env` | |
| Качество | ruff, mypy, pre-commit, GitHub Actions | |

## 5. Структура

```
genius-rag/
  pyproject.toml  docker-compose.yml  Makefile  .env.example
  data/raw/  data/golden/
  notebooks/eda.ipynb
  src/genius_rag/
    config.py
    ingest/      genius_client.py, fetch.py
    store/       pg.py, qdrant.py
    chunking/    docs.py
    embeddings/  encoder.py
    retrieval/   hybrid.py, rerank.py
    llm/         provider.py
    agent/       state.py, tools.py, nodes.py, graph.py
    api/         main.py
    bot/         main.py
  eval/          golden_gen.py, retrieval_eval.py, ragas_eval.py, judge_eval.py
  scripts/       demo_*.py
```

## 6. Roadmap

1. Setup — uv, compose, config, Makefile
2. Ingest — Genius client, artists → songs → referents/annotations, JSONL
3. EDA — Pandas: аннотации/трек, длины, языки, дубли → параметры чанкинга
4. Postgres — схема, загрузка, аналитические SQL
5. Chunking — annotation-doc + контекстный префикс, parent-child
6. Qdrant — dense + sparse upsert, фильтры, поиск
7. Retrieval — hybrid RRF + rerank
8. Generation — grounded ответ, structured output, citations, отказ без evidence
9. Agent — LangGraph: router, tools, grade, retry, multi-hop
10. Eval — golden gen, recall@k/MRR, A/B чанкинг, RAGAS, judge
11. API + bot — FastAPI SSE, aiogram
12. Polish — CI, README (диаграмма, метрики), Langfuse
