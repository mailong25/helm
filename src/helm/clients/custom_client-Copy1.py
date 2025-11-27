import itertools
from typing import List, TypedDict, Dict, Any, Optional

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, GeneratedOutput, Token
from helm.clients.client import CachingClient
import time
import openai
from xai_sdk import Client
from xai_sdk.chat import user, system
from mistralai import Mistral
import os
from google import genai
from google.genai import types

class CustomClient(CachingClient):
    """Simple client for tutorials and for debugging."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
    
    def make_request(self, request: Request) -> RequestResult:
        raw_request: Dict[str, Any] = {
            "provider": request.model.split('/')[0],
            "model": request.model.split('/')[1],
            "prompt": request.prompt,
            "temperature": request.temperature if hasattr(request, 'temperature') else 0.0,
            "top_p": request.top_p,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
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
        return generate_response(raw_request)
    
    def _get_model_for_request(self, request: Request) -> str:
        return request.model_engine

class BaseClient:
    """Base client for provider-specific implementations."""

    def __init__(self, thinking_budget: int = 1024, timeout: int = 300):
        self.thinking_budget = thinking_budget
        self.timeout = timeout
    
    def _validate_request(self, raw_request: Dict[str, Any]) -> None:
        """Validate that required fields are present in the request."""
        required_fields = ["model", "n", "prompt"]
        for field in required_fields:
            if field not in raw_request:
                raise ValueError(f"Missing required field: {field}")
    
    def _format_response(self, completions: List[str], start_time: float) -> Dict[str, Any]:
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }
    
    def generate_response(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement generate_response")

# -------- Provider Clients -------- #

class CustomOpenAIClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_response(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_request(raw_request)
        start_time = time.time()
        
        # Convert prompt to OpenAI messages format
        messages = [{"role": "user", "content": raw_request["prompt"]}]
        
        params = {
            "model": raw_request["model"],
            "messages": messages,
            "n": raw_request["n"],
        }
        
        # Handle temperature
        if "temperature" in raw_request and raw_request["temperature"] is not None:
            params["temperature"] = raw_request["temperature"]
        
        # Handle max_tokens vs max_completion_tokens for o4-mini
        if any(raw_request["model"].startswith(prefix) for prefix in ("o3", "o4", "gpt-5")):
            if "max_tokens" in raw_request and raw_request["max_tokens"]:
                params["max_completion_tokens"] = raw_request["max_tokens"] + 1024
            if "temperature" in params:
                del params['temperature']
        else:
            if "max_tokens" in raw_request and raw_request["max_tokens"]:
                params["max_tokens"] = raw_request["max_tokens"]
        
        # Add other optional parameters
        if "top_p" in raw_request and raw_request["top_p"] is not None:
            params["top_p"] = raw_request["top_p"]

        try:
            response = self.client.chat.completions.create(**params)
            completions = [choice.message.content.strip() if choice.message.content else "" for choice in response.choices]
        except Exception as e:
            print(f"OpenAI error: {e}")
            completions = ["I cannot provide information or assist with requests"] * raw_request["n"]
        
        return self._format_response(completions, start_time)


class CustomGeminiClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)

    def _create_generation_config(self, raw_request: Dict[str, Any]) -> tuple[types.GenerateContentConfig, str]:
        model_name = raw_request["model"]
        base_tokens = raw_request.get("max_tokens", 1024)
        temperature = raw_request.get("temperature", 0.0)
        
        max_tokens = base_tokens * 2 + self.thinking_budget + 128
        thinking_budget = self.thinking_budget
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            temperature=temperature,
            maxOutputTokens=max_tokens
        )
        return config, model_name

    def generate_response(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_request(raw_request)
        start_time = time.time()
        config, model_name = self._create_generation_config(raw_request)
        
        try:
            completions = []
            for _ in range(raw_request["n"]):
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=raw_request["prompt"],
                    config=config
                )
                text = response.text.strip() if response.text else ""
                completions.append(text)
        except Exception as e:
            print(f"Gemini error: {e}")
            completions = ["I cannot provide information or assist with requests"] * raw_request["n"]
        
        return self._format_response(completions, start_time)


class CustomXAIClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        self.client = Client(api_key=api_key, timeout=self.timeout)

    def generate_response(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_request(raw_request)
        start_time = time.time()

        completions = []
        for _ in range(raw_request["n"]):
            try:
                chat = self.client.chat.create(model=raw_request["model"])
                chat.append(user(raw_request["prompt"]))
                response = chat.sample()
                content = response.content.strip() if response.content else ""
                completions.append(content)
            except Exception as e:
                print(f"XAI error: {e}")
                completions.append("I cannot provide information or assist with requests")

        return self._format_response(completions, start_time)


class CustomMistralClient(BaseClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        self.client = Mistral(api_key=api_key)

    def generate_response(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        self._validate_request(raw_request)
        start_time = time.time()
        
        # Convert prompt to Mistral messages format
        messages = [{"role": "user", "content": raw_request["prompt"]}]
        num_completions = raw_request["n"]

        mistral_request = {
            "model": raw_request["model"],
            "messages": messages,
            "n": num_completions,
        }
        
        # Add optional parameters
        if "temperature" in raw_request and raw_request["temperature"] is not None:
            mistral_request["temperature"] = raw_request["temperature"]
        if "max_tokens" in raw_request and raw_request["max_tokens"]:
            mistral_request["max_tokens"] = raw_request["max_tokens"]

        try:
            response = self.client.chat.complete(**mistral_request)
            completions = [choice.message.content.strip() if choice.message.content else "" 
                         for choice in response.choices]
        except Exception as e:
            print(f"Mistral error: {e}")
            completions = ["I cannot provide information or assist with requests"] * num_completions

        return self._format_response(completions, start_time)


# -------- Factory Dispatcher -------- #

def generate_response(raw_request: Dict[str, Any]) -> Dict[str, Any]:
    provider = raw_request["provider"].lower()

    clients = {
        "openai": CustomOpenAIClient,
        "gemini": CustomGeminiClient,
        "xai": CustomXAIClient,
        "mistral": CustomMistralClient,
    }

    if provider not in clients:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(clients.keys())}")

    try:
        client = clients[provider]()
        return client.generate_response(raw_request)
    except Exception as e:
        print(f"Error creating client for provider {provider}: {e}")
        raise