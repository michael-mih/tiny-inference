import json

from language_to_action import generate_plan, load_prompt


def main():
    instruction = load_prompt()
    result = generate_plan(
        instruction,
        max_new_tokens=120,
        do_sample=False,
    )
    print(json.dumps(result["plan"], indent=2))


if __name__ == "__main__":
    main()
