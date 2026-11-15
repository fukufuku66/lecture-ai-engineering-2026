"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、指定したツールを呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests

from app import tools
from app.agent import agent, request_deferred_tool_approvals

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}
IMMEDIATE_TOOLS = BASE_TOOLS - {"save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(IMMEDIATE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert IMMEDIATE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    pending = agent.run_sync(
        "学習記録を保存してください",
        model=TestModel(call_tools=["save_learning_log"]),
    )
    assert isinstance(pending.output, DeferredToolRequests)
    assert not list((tmp_path / "learning_log").glob("*.txt"))

    monkeypatch.setattr("builtins.input", lambda _: "y")
    results = request_deferred_tool_approvals(pending.output)
    assert "保存する学習内容" in capsys.readouterr().out

    resumed = agent.run_sync(
        message_history=pending.all_messages(),
        deferred_tool_results=results,
        model=TestModel(call_tools=[]),
    )
    files = list((tmp_path / "learning_log").glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == pending.output.approvals[0].args_as_dict()["content"]
    assert not isinstance(resumed.output, DeferredToolRequests)


def test_learning_log_is_not_saved_after_denial(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    pending = agent.run_sync(
        "学習記録を保存してください",
        model=TestModel(call_tools=["save_learning_log"]),
    )
    assert isinstance(pending.output, DeferredToolRequests)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    results = request_deferred_tool_approvals(pending.output)
    resumed = agent.run_sync(
        message_history=pending.all_messages(),
        deferred_tool_results=results,
        model=TestModel(call_tools=[]),
    )

    assert not list((tmp_path / "learning_log").glob("*.txt"))
    returns = _parts(resumed, ToolReturnPart)
    assert any("拒否しました" in str(part.content) for part in returns)
