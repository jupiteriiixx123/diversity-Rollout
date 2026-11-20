from agentgym_adapter.reward import compute_score, format_final_score


def test_score():
    for value in [0, 0.25, 1, -0.1]:
        transcript = "Thought:\nAction: turn right\n" + format_final_score(value)
        assert abs(compute_score("babyai", transcript) - value) < 1e-7
    try:
        compute_score("babyai", "missing marker")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing reward did not raise")


if __name__ == "__main__":
    test_score()
    print("reward test: OK")
