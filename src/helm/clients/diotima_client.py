import itertools
from typing import List, TypedDict, Dict, Any

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.clients.client import CachingClient
import time
import openai
import os

class DiotimaClient(CachingClient):
    """Simple client for tutorials and for debugging."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)

    def make_request(self, request: Request) -> RequestResult:
        messages = [{"role": "user", "content": request.prompt}]

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences or None,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        def do_it() -> Dict[str, Any]:
            return self.invoke_model(raw_request)

        cache_key = CachingClient.make_cache_key(raw_request, request)
        response, cached = self.cache.get(cache_key, wrap_request_time(do_it))

        logprob = 0
        completions = [
            GeneratedOutput(
                text=text,
                logprob=logprob,
                tokens=[Token(text=text, logprob=logprob)],
            )
            for text in response["completions"]
        ]

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        return None

    def _get_model_for_request(self, request: Request) -> str:
        return request.model_engine  # Or use logic to map model names if needed

class DiotimaOpenAIClient(DiotimaClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        
        response = client.chat.completions.create(**raw_request)
        
        completions = [choice.message.content.strip() for choice in response.choices]
        
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

# class YourCustomClient(DiotimaClient):
#     def __init__(self, cache_config: CacheConfig):
#         super().__init__(cache_config=cache_config)
        
#     def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        