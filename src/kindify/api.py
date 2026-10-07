"""Optional HTTP API (extra `api`): `uvicorn kindify.api:app`.

Endpoints: `GET /health`, `POST /moderate` {"comment"}, `POST /feedback` {"comment", "rewrite",
"rating", "consent"}. Feedback is stored only when `consent` is true.
"""


from dataclasses import asdict


def create_app(service=None, store=None):
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("install the extra: pip install 'kindify[api]'") from exc
    from kindify.config import Settings
    from kindify.feedback import FeedbackStore
    from kindify.service import ModerationService, load_classifier, make_rewriter

    if service is None or store is None:
        settings = Settings.from_env()
        service = service or ModerationService(lambda: load_classifier(settings.run_dir), make_rewriter(settings),
                                               timeout_s=settings.timeout_s)
        store = store or FeedbackStore(settings.feedback_db, settings.retention_days)

    class Comment(BaseModel):
        comment: str

    class Feedback(BaseModel):
        comment: str
        rewrite: str
        rating: int
        consent: bool = False

    app = FastAPI(title="kindify")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/moderate")
    def moderate(body: Comment):
        if not body.comment.strip():
            raise HTTPException(status_code=422, detail="empty comment")
        return asdict(service.moderate(body.comment))

    @app.post("/feedback")
    def feedback(body: Feedback):
        try:
            stored = store.add_rating(body.comment, body.rewrite, body.rating, body.consent)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"stored": stored}

    return app


def __getattr__(name):  # pragma: no cover - used by `uvicorn kindify.api:app`
    if name == "app":
        return create_app()
    raise AttributeError(name)
