"""Multiple-choice answerer: the application code this evaluation gate tests.

Replace this module with your own prompt construction and answer selection.
The gate imports `choose` from the baseline and candidate versions and compares
their answers on the same questions with the same model.
"""

import torch


def build_prompt(question):
    return f"Question: {question.strip()}\nAnswer:"


@torch.inference_mode()
def choose(model, tokenizer, question, choices):
    """Return the index of the choice with the highest mean token log-probability."""
    prompt_ids = tokenizer(build_prompt(question), add_special_tokens=False).input_ids
    scores = []
    for choice in choices:
        choice_ids = tokenizer(" " + choice.strip(), add_special_tokens=False).input_ids
        if not choice_ids:
            raise ValueError("Answer choice produced no tokens")
        ids = torch.tensor([prompt_ids + choice_ids], device=model.device)
        logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
        targets = torch.tensor(choice_ids, device=logits.device).unsqueeze(1)
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, targets)
        scores.append(token_logprobs.mean().item())
    return max(range(len(choices)), key=scores.__getitem__)
