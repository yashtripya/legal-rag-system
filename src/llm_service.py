"""
LLM Service Module for Legal RAG System.
Handles interaction with local Ollama LLMs with robust error handling and graceful offline fallback.
"""

import json
import requests
from typing import Dict, Any, Optional, List

from config import OLLAMA_BASE_URL, DEFAULT_LLM_MODEL, LLM_TIMEOUT
from src.prompt_templates import LEGAL_RAG_SYSTEM_PROMPT
from src.retriever import RetrievedContext
from src.citation_formatter import LegalCitationFormatter


class LLMServiceError(Exception):
    """Custom exception raised when LLM generation fails."""
    pass


class LocalLegalLLMService:
    """
    Service client for Ollama local LLM runtime.
    Supports local inference, model status checking, and grounded fallback responses.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = DEFAULT_LLM_MODEL,
        timeout: int = LLM_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama service is reachable and lists installed models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                return {
                    "status": "online",
                    "url": self.base_url,
                    "available_models": models,
                    "target_model_installed": any(self.model_name in m for m in models),
                }
        except Exception as e:
            pass

        return {
            "status": "offline",
            "url": self.base_url,
            "error": "Ollama server not reachable at " + self.base_url,
            "available_models": [],
            "target_model_installed": False,
        }

    def generate_answer(
        self,
        query: str,
        retrieved_context: RetrievedContext,
        history: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a grounded legal answer using retrieved passages.
        If Ollama is offline or fails, falls back gracefully to a evidence-only response.
        
        Args:
            query: User's legal question.
            retrieved_context: Retrieved evidence context.
            history: Optional conversation history.
            model_override: Optional model name override.
            
        Returns:
            Dictionary with 'answer', 'citations', 'is_fallback', and 'model_used'.
        """
        model = model_override or self.model_name
        citations = LegalCitationFormatter.format_citations(retrieved_context.passages)

        if retrieved_context.is_empty():
            return {
                "answer": "The provided legal database contains no documents matching your query. Please upload relevant legal texts (Supreme Court judgments, statutes, or constitutional provisions) to query.",
                "citations": [],
                "is_fallback": False,
                "model_used": model,
            }

        # Format history string
        history_str = "None"
        if history:
            formatted_turns = []
            for turn in history[-4:]:  # Last 4 turns
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                formatted_turns.append(f"{role}: {content}")
            history_str = "\n".join(formatted_turns)

        # Build prompt using strict legal grounding template
        prompt = LEGAL_RAG_SYSTEM_PROMPT.format(
            context=retrieved_context.to_formatted_context(),
            history=history_str,
            query=query,
        )

        # Attempt Ollama generation
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for factual legal answers
                    "top_p": 0.9,
                },
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if resp.status_code == 200:
                res_json = resp.json()
                generated_text = res_json.get("response", "").strip()
                if generated_text:
                    return {
                        "answer": generated_text,
                        "citations": citations,
                        "is_fallback": False,
                        "model_used": model,
                    }
            elif resp.status_code == 404:
                fallback_reason = f"Model '{model}' not found in Ollama. Run `ollama pull {model}` to download it."
            else:
                fallback_reason = f"Ollama HTTP {resp.status_code}: {resp.text}"

        except requests.exceptions.ConnectionError:
            fallback_reason = (
                f"Could not connect to Ollama server at `{self.base_url}`. Ensure Ollama is running (`ollama serve`)."
            )
        except requests.exceptions.Timeout:
            fallback_reason = f"Ollama model response timed out after {self.timeout} seconds."
        except Exception as e:
            fallback_reason = f"LLM Generation Exception: {str(e)}"

        # Grounded Fallback Response (Extracts evidence directly when LLM is offline)
        fallback_answer = (
            f"ℹ️ **Local LLM Status:** {fallback_reason}\n\n"
            f"### Grounded Legal Evidence (Direct Document Retrieval)\n"
            f"While the local LLM answer synthesizer is offline, the system has successfully retrieved the "
            f"following **{len(citations)} relevant legal passages** directly matching your query:\n\n"
        )

        for cit in citations:
            fallback_answer += (
                f"**Excerpt from {cit['title']} ({cit['authority']}, Page {cit['page']}):**\n"
                f"> \"{cit['excerpt']}\"\n\n"
            )

        fallback_answer += (
            "📌 *To enable full AI answer synthesis, install and start Ollama locally (`ollama serve` & `ollama pull llama3.2:1b`).*"
        )

        return {
            "answer": fallback_answer,
            "citations": citations,
            "is_fallback": True,
            "fallback_reason": fallback_reason,
            "model_used": "direct-retrieval-fallback",
        }
