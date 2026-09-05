"""Lazy Groq LLM client with model fallbacks. The pipeline still runs without a key."""

from __future__ import annotations

from dataclasses import dataclass

from config import GROQ_MODEL_CANDIDATES, groq_api_key


@dataclass
class LLMResult:
    text: str
    model: str
    used_llm: bool


class ResearchLLM:
    def __init__(self) -> None:
        self._llm = None
        self._candidates: list[str] = []
        self.model_name = "extractive-fallback"
        self.error = ""
        self._init()

    def _init(self) -> None:
        key = groq_api_key()
        if not key:
            self.error = "GROQ_API_KEY not set; using extractive fallback."
            return
        try:
            from langchain_groq import ChatGroq
        except Exception as exc:  # noqa: BLE001
            self.error = f"langchain-groq unavailable: {exc}"
            return

        self._candidates = list(GROQ_MODEL_CANDIDATES)
        model = self._candidates[0]
        self._llm = ChatGroq(model=model, api_key=key, temperature=0.2, max_retries=2)
        self.model_name = model

    @property
    def available(self) -> bool:
        return self._llm is not None

    def complete(self, system: str, human: str) -> LLMResult:
        if not self._llm:
            return LLMResult(text="", model=self.model_name, used_llm=False)

        from langchain_core.messages import HumanMessage, SystemMessage

        remaining = [self.model_name, *[name for name in self._candidates if name != self.model_name]]
        last_error = ""
        for model in remaining:
            try:
                if model != self.model_name:
                    from langchain_groq import ChatGroq

                    self._llm = ChatGroq(
                        model=model,
                        api_key=groq_api_key(),
                        temperature=0.2,
                        max_retries=2,
                    )
                    self.model_name = model
                response = self._llm.invoke(
                    [SystemMessage(content=system), HumanMessage(content=human)]
                )
                text = getattr(response, "content", "") or ""
                return LLMResult(text=text.strip(), model=self.model_name, used_llm=True)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._llm = None
        self.error = last_error or "Could not complete Groq request."
        self.model_name = "extractive-fallback"
        return LLMResult(text="", model=self.model_name, used_llm=False)
