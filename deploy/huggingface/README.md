---
title: Regulatrace
emoji: 📖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Regulatrace

**Maintainer:** [Abdul Aleem Ansari (@Aleem547)](https://github.com/Aleem547).

An upload-based, temporary-session demo of the document question-answering workflow.
Uses local FastEmbed BM25/Qdrant, BGE reranking on ZeroGPU and Groq generation.

Set the private Space Secret `REGULATRACE_GROQ_API_KEY`. Use a Groq Free Plan organization;
usage quotas apply. Keep ZeroGPU selected and Storage Bucket / Dev Mode off.

Upload up to five text-based PDFs, click **Use these PDFs**, then ask a question.
Answers with missing or foreign document/page citations are withheld. Citation
membership is not proof of claim support. Upload authenticity is not verified.
PDF excerpts and questions are sent to Groq; documents are processed on Hugging Face.

Uploads, indexes and conversations are temporary. This profile has no persistent
accounts or saved sessions and does not load the missing original 42-PDF corpus.
Historical benchmark scores from the full application do not evaluate this profile.

The full React/FastAPI source is available at
<https://github.com/Aleem547/Regulatrace>. Original source credit and permission
notes are in `UPSTREAM.md`: adapted from the original code by Faizan0602; the
[upstream repository record](https://api.github.com/repositories/1403548102) is retained.
