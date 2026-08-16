# Third-party notices and provenance

This repository contains original project code. No application source code was copied from Dify, OpenBB, FinRobot, LangChain example applications, or other GitHub projects. Architectural patterns were reimplemented against public interfaces and documentation.

The project depends on third-party libraries listed in `agent/requirements.txt`, `model/requirements.txt`, and `frontend/package.json`. Important runtime components include FastAPI, LangGraph, Pydantic, Qdrant Client, Redis Client, psycopg, FlagEmbedding/BGE-M3, React, Vite, React Markdown and the OpenAI Python SDK. Each dependency remains subject to its own license; downstream distributors must generate a dependency/license inventory for the exact locked release.

Model artifacts are separate from the repository code:

- Qwen3-14B base model: downloaded by the user and governed by its upstream model license.
- BGE-M3 and BGE reranker: downloaded by the user and governed by their upstream model cards/licenses.
- DeepSeek API: remotely accessed under the user's DeepSeek account and service terms.
- This project's SFT LoRA, expanded tokenizer and embedding patch: distributed separately in the project's release under the release metadata and license stated there.

Do not remove upstream copyright or license files from redistributed dependencies or models. Before a public release, run an automated SBOM/license scan against the exact environment and review any copyleft, model-use or data-use obligations.
