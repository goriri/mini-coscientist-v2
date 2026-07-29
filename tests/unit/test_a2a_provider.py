from coscientist.agents import A2AProvider


def test_a2a_provider_removes_exact_and_trailing_prompt_echoes():
    prompt = "Research question: Does a coating improve cycle life?"
    response = '{"question":"Does a coating improve cycle life?"}'

    assert A2AProvider._without_prompt_echo([response, prompt], prompt) == response
    assert (
        A2AProvider._without_prompt_echo([f"{response}\n{prompt}"], prompt) == response
    )
