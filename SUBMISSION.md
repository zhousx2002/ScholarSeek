# Submission Packaging

The competition's `other.zip` archive must stay below 200 MB. Do not compress the project
directory directly: datasets, generated training pairs, Conda environments, Hugging Face caches,
and the full BGE reranker are reproducibility artifacts rather than source submission files.

Build the archive with a strict source whitelist:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission.ps1 `
  -TeamName "团队名称" `
  -ProjectName "ScholarSeek"
```

The generated file is:

```text
submission\团队名称_ScholarSeek_其他.zip
```

It includes the application code, frontend, tests, configuration template, documentation, and
compact offline fallback reranker. It excludes `.env` and therefore never packages API keys.

The primary `BAAI/bge-reranker-base` fine-tuned model is intentionally not included because its
weight file is approximately 1.1 GB. Record its download location, SHA-256 checksum, training
command, and evaluation metrics in the project document. For a fully offline package, export a
separate quantized student reranker under 200 MB and validate its metrics before replacing the
compact fallback.
