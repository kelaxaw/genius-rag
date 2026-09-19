# genius-rag

Agentic RAG над аннотациями [Genius](https://genius.com) API: гибридный поиск на одном Postgres,
LangGraph-агент с инструментами и честная оценка качества (аннотации как ground truth).

Проект собирается в три версии. **v1 — показываемый результат за 3 недели**, v2 — агент и полный eval,
v3 — обвязка. Каждая версия сама по себе законченный артефакт с README и цифрами.

## 1. Что делает

Источник — Genius API: артисты → треки → referents (фрагмент строки) + annotations
(объяснение сообщества) + метаданные (альбом, фиты, продюсеры, сэмплы/каверы).
Полные тексты не хранятся.

**v1**
- `POST /ask` — вопрос → hybrid retrieval → grade → grounded ответ со ссылками на аннотации;
  отказ, если evidence нет
  - «что значит строчка "…" у Oxxxymiron?» → поиск по аннотациям
- eval retrieval: recall@k, MRR, nDCG; A/B чанкинга; dense vs hybrid → таблица метрик в README

**v2**
- `POST /ask` становится агентом: роутер → `vector_search | sql_query | song_graph` → rerank → grade → answer, retry при low coverage
  - «сколько треков у Скриптонита с фитами Truwer?» → sql_query
  - «кого сэмплирует трек X и что об этом пишут?» → song_graph + vector_search (multi-hop)
- `POST /annotate` — строка без аннотации → черновик объяснения (few-shot из retrieval)
- eval generation: RAGAS faithfulness/relevancy, LLM-judge «черновик vs реальная аннотация»

**v3**
- Telegram-бот (aiogram 3) поверх `/ask`, Langfuse-трейсы, CI

## 2. Не делает

- полные тексты песен (копирайт), UI, авторизация
- весь Genius — **v1: 6–8 артистов (ru + en), ~1–2k аннотаций**; v2: 10–20 артистов, 2–5k
- отдельный векторный движок — всё в Postgres (см. §7)

## 3. Архитектура

```
Genius API ─> ingest (httpx, rate-limit) ─> data/raw/*.jsonl ─> Pandas EDA
                                              └─> Postgres 16
                                                    ├─ artists / songs / song_relationships / annotations
                                                    └─ chunks: text, tsvector (BM25-подобный), embedding vector(384)

v1 pipeline:
  question ─> hybrid (dense kNN ∪ full-text, RRF) ─> grade ─> answer(+citations) | refuse

v2 agent (LangGraph):
  route ─> [vector_search | sql_query | song_graph] ─> rerank(опц.) ─> grade ─> answer(+citations)
                                  ▲                                     │ low coverage
                                  └────────────── retry ────────────────┘

FastAPI /ask (SSE), /annotate   ·   eval/   ·   v3: aiogram bot, Langfuse
```

## 4. Стек

| Слой | Выбор | Почему |
|---|---|---|
| Язык | Python 3.12+, `uv` | стандарт |
| Оркестрация | v1: обычные функции; v2: LangChain + LangGraph, tool-calling | маршрут зависит от вопроса |
| LLM | OpenRouter via `langchain-openai`; GigaChat опц. | абстракция провайдера |
| Embeddings | `intfloat/multilingual-e5-small` + contextual prefix | ru+en, CPU, 384 dim |
| Sparse | Postgres full-text (`tsvector`, `ts_rank_cd`, ru+en конфиги) | hybrid без второго движка |
| Vector store | Postgres 16 + `pgvector` (HNSW, cosine) | один сервис, один compose |
| Метаданные | тот же Postgres — связи треков, text-to-SQL tool (v2) | |
| Reranker | `BAAI/bge-reranker-v2-m3`, за флагом `RERANK=1` (v2) | измерить прирост, не включать по умолчанию |
| Данные | Genius API → JSONL, Pandas для EDA | |
| API | FastAPI + SSE | REST/JSON |
| Чат | aiogram 3 (v3) | |
| Eval | pytest + свой harness (v1) + RAGAS + LLM-judge (v2) | ground truth = аннотации |
| Observability | Langfuse (v3, docker) | трейсы агента |
| Инфра | Docker Compose (app, postgres), Makefile, `.env` | |
| Качество | ruff, mypy, pre-commit; GitHub Actions (v3) | |

## 5. Структура

```
genius-rag/
  pyproject.toml  docker-compose.yml  Makefile  .env.example  README.md
  data/raw/  data/golden/
  notebooks/eda.ipynb
  src/genius_rag/
    config.py
    ingest/      genius_client.py, fetch.py
    db/          pg.py (схема, миграции, upsert), search.py (kNN, fulltext, RRF в SQL)
    chunking/    docs.py
    embeddings/  encoder.py
    retrieval/   hybrid.py, rerank.py (v2)
    llm/         provider.py
    generation/  answer.py (grounded ответ, citations, refuse)   # v1
    agent/       state.py, tools.py, nodes.py, graph.py           # v2
    api/         main.py
    bot/         main.py                                          # v3
  eval/          golden_gen.py, retrieval_eval.py, ragas_eval.py (v2), judge_eval.py (v2)
  scripts/       demo_ask.py
```

## 6. Roadmap

### v1 — до 9 октября (~40 ч)

| # | Шаг | Часы | Готово когда |
|---|---|---|---|
| 1 | Setup — uv, compose (app + postgres/pgvector), config, Makefile | ✓ | `make up` поднимает всё |
| 2 | Ingest — client, artists → songs → referents/annotations, JSONL | ✓ | 6–8 артистов в `data/raw` |
| 3 | EDA — аннотации/трек, длины, языки, дубли | 3 | параметры чанкинга выбраны и записаны в README |
| 4 | Postgres — схема, миграции, загрузка | 4 | `SELECT count(*)` по всем таблицам, 3 аналитических запроса |
| 5 | Chunking — annotation-doc + контекстный префикс (артист / трек / строка), parent-child | 4 | таблица `chunks` |
| 6 | Embeddings + индексы — e5-small, `vector(384)` HNSW, `tsvector` + GIN | 5 | kNN и full-text по отдельности работают |
| 7 | Hybrid — RRF в одном SQL, payload-фильтры (артист, язык) | 5 | `search(q, k)` возвращает chunks с обоими скорами |
| 8 | Generation — grounded ответ, structured output, citations, отказ без evidence | 6 | `/ask` через FastAPI + SSE, `scripts/demo_ask.py` |
| 9 | Eval retrieval — golden set (LLM-генерация вопросов из аннотаций + ручная правка 50–100), recall@k / MRR / nDCG; A/B: чанкинг × dense-only vs hybrid | 8 | таблица метрик |
| 10 | README — диаграмма, таблица метрик, 3 примера вопрос → ответ, как поднять | 4 | репозиторий можно скинуть по ссылке |

### v2 — до 30 октября (~35 ч)

| # | Шаг | Часы |
|---|---|---|
| 11 | LangGraph — state, router, три тула (`vector_search`, `sql_query`, `song_graph`), grade, retry | 12 |
| 12 | Multi-hop вопросы (сэмплы/каверы через `song_relationships`) | 4 |
| 13 | Reranker за флагом; измерить прирост на golden | 4 |
| 14 | `/annotate` — few-shot черновик | 3 |
| 15 | Eval generation — RAGAS, LLM-judge черновик vs аннотация | 8 |
| 16 | Расширить до 10–20 артистов, перегнать метрики | 4 |

### v3 — если есть время

17. Telegram-бот (aiogram) · 18. GitHub Actions (ruff, mypy, pytest) · 19. Langfuse

## 7. Решения

- **pgvector + tsvector вместо Qdrant.** Один сервис вместо двух, hybrid и фильтры делаются одним SQL,
  сравнение dense/hybrid — тот же запрос с флагом. На 2–5k аннотаций разницы в качестве и скорости нет;
  Qdrant вернётся только если объём вырастет на порядок.
- **v1 без LangGraph.** Линейный pipeline проще отладить и измерить; агент добавляется поверх
  готового и уже измеренного retrieval, а не вместе с ним.
- **Reranker — опция, не дефолт.** Его ценность надо показать цифрой на golden, а не заявить.
- **Golden set частично ручной.** 50–100 вопросов, проверенных глазами, весят на собесе больше, чем
  500 сгенерированных.

## 8. Что README должен отвечать

- почему такой чанкинг (A/B с цифрами)
- что даёт hybrid vs dense (цифры)
- сколько вопросов уходит в отказ и почему это правильно
- v2: где агент выбирает SQL, а где вектор, и сколько раз срабатывает retry
