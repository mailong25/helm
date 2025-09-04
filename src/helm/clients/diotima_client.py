import itertools
from typing import List, TypedDict, Dict, Any

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

class DiotimaClient(CachingClient):
    """Simple client for tutorials and for debugging."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)

    def make_request(self, request: Request) -> RequestResult:
        messages = [{"role": "user", "content": request.prompt}]
        #print(request)
        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": 0.0,
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
        if 'o4-mini' in raw_request["model"]:
            raw_request['max_completion_tokens'] = max(raw_request['max_tokens'] + 1024, 5024)
            del raw_request['max_tokens']
            del raw_request['temperature']
        
        response = client.chat.completions.create(**raw_request)
        
        completions = [choice.message.content.strip() for choice in response.choices]
        
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

class DiotimaGeminiClient(DiotimaClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.thinking_budget = 2048
    
    def _create_generation_config(self, raw_request: Dict[str, Any]) -> tuple[types.GenerateContentConfig, str]:
        """Create generation config and determine model name based on request."""
        model_name = raw_request["model"]
        base_tokens = raw_request["max_tokens"]
        temperature = raw_request["temperature"]
        
        if "-thinking" in model_name:
            max_tokens = base_tokens * 2 + self.thinking_budget + 128
            #max_tokens = 8192
            thinking_budget = self.thinking_budget
            model_name = model_name.replace("-thinking", "")
        else:
            max_tokens = base_tokens
            thinking_budget = 0
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            temperature=temperature,
            maxOutputTokens=max_tokens
        )
        
        return config, model_name
    
    def _generate_completions(self, client: genai.Client, model_name: str, 
                            content: str, config: types.GenerateContentConfig, 
                            num_completions: int) -> List[str]:
        """Generate completions from the Gemini API."""
        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=content,
                config=config
            )
            
            if not isinstance(response.text, str) or len(response.text) == 0:
                raise ValueError("Response is not a valid non-empty string")
            
            return [response.text] * num_completions
        
        except Exception as e:
            print(f"Failed to generate content: {e}")
            return ["I cannot provide information or assist with requests"] * num_completions
    
    def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        client = genai.Client(api_key=self.gemini_api_key)
        config, model_name = self._create_generation_config(raw_request)
        
        completions = self._generate_completions(
            client=client,
            model_name=model_name,
            content=raw_request["messages"][0]["content"],
            config=config,
            num_completions=raw_request["n"]
        )
        
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

class DiotimaXAIClient(DiotimaClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    
    def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        client = Client(
            api_key=self.XAI_API_KEY,
            timeout=300,
        )
        
        user_prompt = raw_request["messages"][0]["content"]
        completions = []
        num_completions = raw_request.get("n", 1)
        
        for _ in range(num_completions):
            chat = client.chat.create(model=raw_request["model"])
            chat.append(user(user_prompt))
            try:
                response = chat.sample()
                completions.append(response.content.strip())
            except Exception as e:
                print(f'Failed test: {e}')
                completions.append("I cannot provide information or assist with requests")
        
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

class DiotimaMistralClient(DiotimaClient):
    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    
    def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        client = Mistral(api_key=self.MISTRAL_API_KEY)
        
        messages = [{"role": "user", "content": raw_request["messages"][0]["content"]}]

        num_completions = raw_request.get("n", 1)
        
        mistral_request = {
            "model": raw_request["model"],
            "messages": messages,
            "temperature": raw_request['temperature'],
            "n": num_completions,
        }
        
        if raw_request.get("max_tokens"):
            mistral_request["max_tokens"] = raw_request["max_tokens"]
        
        completions = []
        response = client.chat.complete(**mistral_request)
        
        for i in range(num_completions):
            try:
                completions.append(response.choices[i].message.content.strip())
            except Exception as e:
                print(f'Failed test: {e}')
                completions.append("I cannot provide information or assist with requests")
        
        return {
            "completions": completions,
            "request_time": time.time() - start_time,
            "request_datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        }

# class YourCustomClient(DiotimaClient):
#     def __init__(self, cache_config: CacheConfig):
#         super().__init__(cache_config=cache_config)
        
#     def invoke_model(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        