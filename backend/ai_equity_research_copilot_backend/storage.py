from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

from .schemas import (
    Company,
    CompanyCreate,
    Conversation,
    Document,
    DocumentChunk,
    DocumentCreate,
    DocumentStatus,
    Message,
    StoredCitation,
    utcnow,
)


EMPTY_STATE: dict[str, list[dict[str, Any]]] = {
    "companies": [],
    "documents": [],
    "chunks": [],
    "conversations": [],
    "messages": [],
    "citations": [],
}


class JsonRepository:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._lock = RLock()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self._write_state(EMPTY_STATE.copy())

    def health_counts(self) -> dict[str, int]:
        state = self._read_state()
        return {key: len(state[key]) for key in ["companies", "documents", "chunks"]}

    def list_companies(self, search: str | None = None) -> list[Company]:
        state = self._read_state()
        companies = [Company.model_validate(item) for item in state["companies"]]
        if search:
            needle = search.lower()
            companies = [
                company
                for company in companies
                if needle in company.ticker.lower() or needle in company.name.lower()
            ]
        return sorted(companies, key=lambda company: company.ticker)

    def create_company(self, payload: CompanyCreate) -> Company:
        with self._lock:
            state = self._read_state()
            ticker = payload.ticker.upper()
            if any(item["ticker"].upper() == ticker for item in state["companies"]):
                raise ValueError(f"Company ticker '{ticker}' already exists")
            company = Company(**payload.model_dump())
            state["companies"].append(company.model_dump(mode="json"))
            self._write_state(state)
            return company

    def upsert_company(self, payload: CompanyCreate) -> Company:
        existing = next((company for company in self.list_companies() if company.ticker == payload.ticker), None)
        if existing:
            return existing
        return self.create_company(payload)

    def get_company(self, company_id: UUID) -> Company | None:
        return self._get_model("companies", company_id, Company)

    def list_documents(self, company_id: UUID | None = None) -> list[Document]:
        state = self._read_state()
        documents = [Document.model_validate(item) for item in state["documents"]]
        if company_id:
            documents = [document for document in documents if document.company_id == company_id]
        return sorted(documents, key=lambda document: document.created_at)

    def create_document(self, company_id: UUID, payload: DocumentCreate, file_path: str | None) -> Document:
        if not self.get_company(company_id):
            raise KeyError(f"Company '{company_id}' not found")
        document = Document(company_id=company_id, file_path=file_path, **payload.model_dump())
        with self._lock:
            state = self._read_state()
            state["documents"].append(document.model_dump(mode="json"))
            self._write_state(state)
        return document

    def get_document(self, document_id: UUID) -> Document | None:
        return self._get_model("documents", document_id, Document)

    def update_document(self, document: Document) -> Document:
        document.updated_at = utcnow()
        self._replace_model("documents", document)
        return document

    def delete_document(self, document_id: UUID) -> bool:
        with self._lock:
            state = self._read_state()
            before = len(state["documents"])
            state["documents"] = [item for item in state["documents"] if item["id"] != str(document_id)]
            state["chunks"] = [item for item in state["chunks"] if item["document_id"] != str(document_id)]
            self._write_state(state)
            return len(state["documents"]) < before

    def replace_chunks(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        with self._lock:
            state = self._read_state()
            state["chunks"] = [item for item in state["chunks"] if item["document_id"] != str(document_id)]
            state["chunks"].extend(chunk.model_dump(mode="json") for chunk in chunks)
            self._write_state(state)

    def list_chunks(
        self,
        document_id: UUID | None = None,
        company_ids: list[UUID] | None = None,
    ) -> list[DocumentChunk]:
        state = self._read_state()
        chunks = [DocumentChunk.model_validate(item) for item in state["chunks"]]
        if document_id:
            chunks = [chunk for chunk in chunks if chunk.document_id == document_id]
        if company_ids:
            company_set = set(company_ids)
            chunks = [chunk for chunk in chunks if chunk.company_id in company_set]
        return sorted(chunks, key=lambda chunk: (str(chunk.document_id), chunk.chunk_index))

    def create_conversation(self, title: str) -> Conversation:
        conversation = Conversation(title=title[:80] or "Research conversation")
        with self._lock:
            state = self._read_state()
            state["conversations"].append(conversation.model_dump(mode="json"))
            self._write_state(state)
        return conversation

    def list_conversations(self) -> list[Conversation]:
        state = self._read_state()
        conversations = [Conversation.model_validate(item) for item in state["conversations"]]
        return sorted(conversations, key=lambda conversation: conversation.updated_at, reverse=True)

    def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        return self._get_model("conversations", conversation_id, Conversation)

    def create_message(self, message: Message) -> Message:
        with self._lock:
            state = self._read_state()
            state["messages"].append(message.model_dump(mode="json"))
            self._write_state(state)
        return message

    def list_messages(self, conversation_id: UUID | None = None) -> list[Message]:
        state = self._read_state()
        messages = [Message.model_validate(item) for item in state["messages"]]
        if conversation_id:
            messages = [message for message in messages if message.conversation_id == conversation_id]
        return sorted(messages, key=lambda message: message.created_at)

    def create_citations(self, citations: list[StoredCitation]) -> None:
        with self._lock:
            state = self._read_state()
            state["citations"].extend(citation.model_dump(mode="json") for citation in citations)
            self._write_state(state)

    def list_citations(self, message_id: UUID | None = None) -> list[StoredCitation]:
        state = self._read_state()
        citations = [StoredCitation.model_validate(item) for item in state["citations"]]
        if message_id:
            citations = [citation for citation in citations if citation.message_id == message_id]
        return sorted(citations, key=lambda citation: citation.created_at)

    def _get_model(self, key: str, item_id: UUID, model: type[Any]) -> Any | None:
        state = self._read_state()
        for item in state[key]:
            if item["id"] == str(item_id):
                return model.model_validate(item)
        return None

    def _replace_model(self, key: str, model: Any) -> None:
        with self._lock:
            state = self._read_state()
            model_id = str(model.id)
            for index, item in enumerate(state[key]):
                if item["id"] == model_id:
                    state[key][index] = model.model_dump(mode="json")
                    self._write_state(state)
                    return
            raise KeyError(f"{key} item '{model_id}' not found")

    def _read_state(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            if not self.state_file.exists():
                return EMPTY_STATE.copy()
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            return {key: list(raw.get(key, [])) for key in EMPTY_STATE}

    def _write_state(self, state: dict[str, list[dict[str, Any]]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_file)
