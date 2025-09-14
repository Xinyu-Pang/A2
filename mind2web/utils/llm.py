import logging
import re
import os
import inspect
import tiktoken
from openai import OpenAI

logger = logging.getLogger("main")

class Caller():
    """Run openai api to generate response."""
    def __init__(self, model):
        self.client = OpenAI(api_key="YOUR_API_KEY")
        self.model = model
        self.MAX_TOKENS = {
            "GPT-3-5-turbo-chat": 4097,
            "gpt-3.5-turbo-0301": 4097,
            "gpt-3.5-turbo-0613": 4097,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-4": 8192,
            "gpt-4o": 16385,
            "gpt-4o-mini":128000,
            "GPT-3-5-16k-turbo-chat": 16385,
            "gpt-4-32k": 32000,
            "gpt-3.5-turbo": 16385,
            "gpt-35-turbo": 8192,
            "deepseek-r1": 57344,
            "deepseek-v3": 57344
        }
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.cal_tokens = 0
    
    def call(self, messages, temperature=0.0, n=1, stop_tokens=None):
        try:
            response = self.client.chat.completions.create(
                    messages = messages,
                    model = self.model,
                    n=n,
                    temperature= 0.0,
                    stop=stop_tokens if stop_tokens else None)
            self.completion_tokens += response.usage.completion_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.total_tokens += response.usage.total_tokens
            if n == 1:
                p = response.choices[0].message.content.strip()
            else:
                p = [choice.message.content.strip() for choice in response.choices]
            return p
        except Exception as e:
            try:
                if response.choices[0].message.content is None:
                    return ""
            except:
                return ""
    
    def get_embedding(self, message):
        """Return the embedding of a message."""
        try:
            encoding = self.client.embeddings.create(input=message, model=self.model, encoding_format="float").data[0].embedding
            return encoding
        except Exception as e:
            print(e)
            return None

    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "GPT-3-5-turbo-chat",
            "GPT-3-5-16k-turbo-chat",
            "gpt-3.5-16k-turbo-chat",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
            "gpt-4o-mini",
            "deepseek-r1",
            "deepseek-v3"
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_mode(self, model: str) -> str:
        """Check if the model is a chat model."""
        if model in [
            "GPT-3-5-turbo-chat",
            "GPT-3-5-16k-turbo-chat",
            "gpt-3.5-16k-turbo-chat",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4",
            "gpt-4o",
            "gpt-4-32k-0613",
        ]:
            return "chat"
        elif model in [
            "davinci-002",
            "gpt-3.5-turbo-instruct-0914",
        ]:
            return "completion"
        else:
            raise ValueError(f"Unknown model: {model}")

    def extract_from_response(self, response: str, backtick="```") -> str:
        if backtick == "```":
            # Matches anything between ```<optional label>\n and \n```
            pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
        elif backtick == "`":
            pattern = r"`(.*?)`"
        else:
            raise ValueError(f"Unknown backtick: {backtick}")
        match = re.search(
            pattern, response, re.DOTALL
        )  # re.DOTALL makes . match also newlines
        if match:
            extracted_string = match.group(1)
        else:
            extracted_string = ""

        return extracted_string

    def compute_cost(self):
        if "gpt-4o-mini" in self.model:
            input_price = 0.15/1000
            output_price = 0.6/1000
            embedding_price = 0.0001
        else:
            raise ValueError(f"Unknown model: {self.model}")
        rate = 1
        cost = float(self.prompt_tokens) / 1000 * input_price * rate + \
               float(self.completion_tokens) / 1000 * output_price * rate 
        return "{:.2f}".format(cost)
