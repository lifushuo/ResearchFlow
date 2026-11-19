"""观测固定版本 ChatOpenAI 的响应转换钩子，不替换传输或解析器。"""

from langchain_openai import ChatOpenAI

from research_engine.response_diagnostics import current_observation, observe


class _ObservedRawResponse:
    def __init__(self, response):
        self._response = response

    def __getattr__(self, name):
        return getattr(self._response, name)

    def parse(self, *args, **kwargs):
        try:
            return self._response.parse(*args, **kwargs)
        except BaseException as exc:
            observe("conversion_failed", exc)
            raise


class _ObservedRawCompletions:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def create(self, *args, **kwargs):
        response = await self._client.create(*args, **kwargs)
        if current_observation() is None:
            return response
        observe("http_response", getattr(response, "http_response", None))
        return _ObservedRawResponse(response)


class _ObservedCompletions:
    def __init__(self, client):
        self._client = client
        self.with_raw_response = _ObservedRawCompletions(client.with_raw_response)

    def __getattr__(self, name):
        return getattr(self._client, name)


class ObservedChatOpenAI(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.async_client = _ObservedCompletions(self.async_client)

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        # 读取 SDK 重命名及覆盖后的实际请求参数。密钥仅用于脱敏匹配，绝不进入导出观测。
        observe("request_options", payload, self.openai_api_key)
        return payload

    def _create_chat_result(self, response, generation_info=None):
        observe("before_conversion", response)
        try:
            result = super()._create_chat_result(response, generation_info)
        except BaseException as exc:
            observe("conversion_failed", exc)
            raise
        if result.generations:
            observe("after_conversion", result.generations[0].message)
        return result
