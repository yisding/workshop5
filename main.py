from openai import OpenAI

client = OpenAI()

messages = []

print("Chat started. Type 'exit' or 'quit' to stop.")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
    )

    assistant_reply = response.choices[0].message.content or ""
    print(f"Assistant: {assistant_reply}")

    messages.append({"role": "assistant", "content": assistant_reply})