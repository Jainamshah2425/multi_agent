import os
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import openai
import requests
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from config import config
from utils import logger

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        pass

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"

class MistralLLM(BaseLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": "mistral-small",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Mistral API error: {str(e)}")
            return f"Error: {str(e)}"

class GoogleLLM(BaseLLM):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens))
            return response.text
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            return f"Error: {str(e)}"

class LocalLLM(BaseLLM):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            # Truncate prompt if it exceeds model's max input length
            max_input_length = self.tokenizer.model_max_length  # Get model's max input length
            if len(prompt) > max_input_length:
                logger.warning(f"Prompt length ({len(prompt)}) exceeds model's max input length ({max_input_length}). Truncating prompt.")
                prompt = prompt[:max_input_length]

            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # Extract only the generated text, excluding the input prompt
            generated_text = result[0]['generated_text']
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            return f"Error: {str(e)}"

class LLMManager:
    def __init__(self):
        self._llm_cache = {}
    
    def get_llm(self, llm_type: str) -> BaseLLM:
        if llm_type in self._llm_cache:
            return self._llm_cache[llm_type]
        
        if llm_type == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            llm = OpenAILLM(config.OPENAI_API_KEY)
        elif llm_type == "mistral":
            if not config.MISTRAL_API_KEY:
                raise ValueError("Mistral API key not found")
            llm = MistralLLM(config.MISTRAL_API_KEY)
        elif llm_type == "google":
            if not config.GOOGLE_API_KEY:
                raise ValueError("Google API key not found")
            llm = GoogleLLM(config.GOOGLE_API_KEY)
        elif llm_type == "local":
            llm = LocalLLM()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        self._llm_cache[llm_type] = llm
        return llm


llm_manager = LLMManager()
