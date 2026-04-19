from bughound_agent import BugHoundAgent
from llm_client import MockClient


class EmptyJsonAnalyzerClient:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "Return ONLY valid JSON" in system_prompt:
            return "[]"
        return "# no-op rewrite\n"


def test_workflow_runs_in_offline_mode_and_returns_shape():
    agent = BugHoundAgent(client=None)  # heuristic-only
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert isinstance(result, dict)
    assert "issues" in result
    assert "fixed_code" in result
    assert "risk" in result
    assert "logs" in result

    assert isinstance(result["issues"], list)
    assert isinstance(result["fixed_code"], str)
    assert isinstance(result["risk"], dict)
    assert isinstance(result["logs"], list)
    assert len(result["logs"]) > 0


def test_offline_mode_detects_print_issue():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") ==
               "Code Quality" for issue in result["issues"])


def test_offline_mode_proposes_logging_fix_for_print():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    fixed = result["fixed_code"]
    assert "logging" in fixed
    assert "logging.info(" in fixed


def test_mock_client_forces_llm_fallback_to_heuristics_for_analysis():
    # MockClient returns non-JSON for analyzer prompts, so agent should fall back.
    agent = BugHoundAgent(client=MockClient())
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") ==
               "Code Quality" for issue in result["issues"])
    # Ensure we logged the fallback path
    assert any("Falling back to heuristics" in entry.get("message", "")
               for entry in result["logs"])


def test_empty_llm_issue_list_falls_back_to_heuristics_when_code_has_obvious_issue():
    agent = BugHoundAgent(client=EmptyJsonAnalyzerClient())
    code = "def load_text_file(path):\n    try:\n        f = open(path, 'r')\n        data = f.read()\n        f.close()\n    except:\n        return None\n\n    return data\n"
    result = agent.run(code)

    assert any(issue.get("type") ==
               "Reliability" for issue in result["issues"])
    assert any("LLM returned no issues" in entry.get("message", "")
               for entry in result["logs"])
