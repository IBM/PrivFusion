import logging
import os
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, cast

from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain_core.outputs import ChatGeneration
from langchain_ibm.chat_models import ChatWatsonx
from langchain_ollama import ChatOllama
from litellm import NotFoundError, RateLimitError, Timeout, completion
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLM(ABC):
    @abstractmethod
    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        raise NotImplementedError()


class WatsonXLLM(LLM):
    def __init__(self, model_name: str):
        load_dotenv()

        parameters = {
            # GenTextParamsMetaNames.DECODING_METHOD: "sample",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 500,
            GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
            GenTextParamsMetaNames.TEMPERATURE: 0.5,
            GenTextParamsMetaNames.TOP_K: 50,
            GenTextParamsMetaNames.TOP_P: 1,
        }

        self._model_name = model_name
        self._model = ChatWatsonx(
            model_id=self._model_name,
            url=SecretStr("https://us-south.ml.cloud.ibm.com"),
            apikey=SecretStr(os.environ["WATSONX_APIKEY"]),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            params=parameters,
        )

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        response: dict[str, str] = {}
        try:
            result = self._model.generate([messages], **kwargs)
            print(">>>>")
            pprint(result)
            print("<<<<")
            logger.info(result)
            messages = result.generations[-1]
            last_message = cast(ChatGeneration, messages[-1]).message
            response["response"] = str(last_message.content)

        except Timeout as e:
            print(f"WatsonX time out error. Check Metrics Dashboard:\n{e.message}")

        except RateLimitError as e:
            print(
                f"WatsonX user rate limit (8 requests per second) is exceeded\n{e.message}",
            )

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(e)

        return response


class RitsLLM(LLM):
    def __init__(self, model_endpoint: str, model_name: str):
        self._api_key = os.getenv("RITS_API_KEY", None)
        if self._api_key is None:
            raise OSError("RITS_API_KEY not found in environment.")
        self._model_endpoint = model_endpoint
        self._model_name = model_name

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            completion_response = completion(
                model="openai/" + self._model_name,
                api_key="fake-key",
                messages=messages,
                api_base=self._model_endpoint + "/v1",
                extra_headers={"RITS_API_KEY": self._api_key},
                **kwargs,
            )
            response = completion_response.choices[0].message.content
            reason = completion_response.choices[0].message.reasoning_content
            result["response"] = response
            result["reason"] = reason

        except Timeout as e:
            print(f"RITS time out error. Check Metrics Dashboard:\n{e.message}")

        except RateLimitError as e:
            print(
                f"RITS user rate limit (1500 requests per minute) is exceeded\n{e.message}",
            )

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(e)

        return result


class OllamaLLM(LLM):
    def __init__(
        self,
        model_name: str,
        temperature: int = 0,
        max_tokens: int = 1000,
        reasoning: bool = True,
    ):
        self._model_name = model_name

        self._model = ChatOllama(
            model=self._model_name,
            temperature=temperature,
            num_predict=max_tokens,
        )

    def chat(self, messages: list[Any], **kwargs: Any) -> dict[str, str]:
        reasoning = kwargs.get("reasoning", False)
        result: dict[str, str] = {}
        try:
            response = self._model.invoke(messages, reasoning=reasoning)
            result["response"] = response.content
            result["reason"] = response.additional_kwargs["reasoning_content"]

        except Timeout as e:
            print(f"Ollama time out error:\n{e.message}")

        # except RateLimitError as e:
        #     print(f"RITS user rate limit (1500 requests per minute) is exceeded\n{e.message}")

        except NotFoundError as e:
            print(f"Selected model not found: {self._model_name}\n{e.message}")

        except Exception as e:
            print(">>>> EXCEPTION:")
            print(e)

        return result
