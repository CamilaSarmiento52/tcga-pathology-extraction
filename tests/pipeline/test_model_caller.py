from unittest.mock import MagicMock, patch

from src.pipeline.model_caller import ModelResponse, call_model


def _mock_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


class TestModelResponse:
    def test_dataclass_fields(self):
        r = ModelResponse(
            raw_text="x", input_tokens=10, output_tokens=5, latency_ms=200.0, model="openai:gpt-4o"
        )
        assert r.raw_text == "x"
        assert r.input_tokens == 10
        assert r.latency_ms == 200.0
        assert r.model == "openai:gpt-4o"


class TestOpenAIStructuredPath:
    @patch("src.pipeline.model_caller.OpenAI")
    def test_uses_parse_with_response_format(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.parse.return_value = _mock_completion(
            '{"primary_site": "left breast"}'
        )

        result = call_model("prompt", model="openai:o4-mini")

        assert isinstance(result, ModelResponse)
        assert client.chat.completions.parse.call_count == 1
        # The structured path never falls back to plain create / no retry needed.
        assert client.chat.completions.create.call_count == 0
        kwargs = client.chat.completions.parse.call_args.kwargs
        assert kwargs["response_format"].__name__ == "LLMExtraction"
        assert kwargs["model"] == "o4-mini"  # provider prefix stripped
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model == "openai:o4-mini"

    @patch("src.pipeline.model_caller.OpenAI")
    def test_reasoning_model_omits_temperature(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.parse.return_value = _mock_completion("{}")

        call_model("p", model="openai:o4-mini")

        kwargs = client.chat.completions.parse.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["max_completion_tokens"] == 4096


class TestOllamaOpenAISDKPath:
    """Ollama calls now go through the OpenAI SDK (via /v1) so mlflow.openai.autolog()
    can capture them. num_ctx is passed via extra_body["options"]."""

    @patch("src.pipeline.model_caller.OpenAI")
    def test_no_retry_on_valid_json(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion(
            '{"primary_site": "left breast"}', prompt_tokens=100, completion_tokens=50
        )

        result = call_model("p", model="ollama:mistral:7b")

        assert client.chat.completions.create.call_count == 1
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @patch("src.pipeline.model_caller.OpenAI")
    def test_retries_once_on_unparseable(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_completion("Sorry, I cannot help with that."),
            _mock_completion('{"primary_site": "x"}'),
        ]

        result = call_model("p", model="ollama:llama3.2")

        assert client.chat.completions.create.call_count == 2
        assert '{"primary_site": "x"}' in result.raw_text

    @patch("src.pipeline.model_caller.OpenAI")
    def test_retries_on_schema_invalid_with_field_correction(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _mock_completion('{"tnm_stage": {"T": 42}}'),
            _mock_completion('{"tnm_stage": {"T": "pT2"}}'),
        ]

        result = call_model("p", model="ollama:mistral:7b")

        assert client.chat.completions.create.call_count == 2
        second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        second_prompt = second_messages[0]["content"]
        assert "tnm_stage.T" in second_prompt
        assert "pT2" in result.raw_text

    @patch("src.pipeline.model_caller.OpenAI")
    def test_strips_provider_prefix_for_local(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"primary_site": "x"}')

        call_model("p", model="ollama:mistral:7b")

        assert client.chat.completions.create.call_args.kwargs["model"] == "mistral:7b"

    @patch("src.pipeline.model_caller.OpenAI")
    def test_passes_json_format_in_extra_body(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"primary_site": "x"}')

        call_model("p", model="ollama:mistral:7b")

        extra_body = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert extra_body.get("format") == "json"

    @patch("src.pipeline.model_caller.OpenAI")
    def test_num_ctx_in_options_when_set(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"primary_site": "x"}')

        call_model("p", model="ollama:mistral:7b", num_ctx=16384)

        extra_body = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert extra_body["options"]["num_ctx"] == 16384

    @patch("src.pipeline.model_caller.OpenAI")
    def test_num_ctx_omitted_when_none(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"primary_site": "x"}')

        call_model("p", model="ollama:mistral:7b")

        extra_body = client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert "options" not in extra_body

    @patch("src.pipeline.model_caller.OpenAI")
    def test_uses_v1_base_url(self, mock_openai_cls):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _mock_completion('{"primary_site": "x"}')

        call_model("p", model="ollama:mistral:7b")

        base_url = mock_openai_cls.call_args.kwargs.get("base_url") or mock_openai_cls.call_args.args[0] if mock_openai_cls.call_args.args else mock_openai_cls.call_args.kwargs.get("base_url")
        assert base_url is None or "/v1" in str(base_url)
