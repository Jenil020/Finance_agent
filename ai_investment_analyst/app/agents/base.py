"""Base class / shared utilities for all agents."""
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings
from app.core.logging import logger

genai.configure(api_key=settings.google_api_key)


def get_gemini_model(model_name: str = "gemini-1.5-flash"):
    """Return a configured Gemini model instance."""
    return genai.GenerativeModel(model_name)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def llm_call_with_retry(model, prompt: str) -> str:
    """LLM call with exponential backoff retry (Tenacity)."""
    response = await model.generate_content_async(prompt)
    return response.text
