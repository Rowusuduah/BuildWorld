"""
Basic usage examples for context-trim.

Shows the common patterns for trimming conversation histories.
"""

from context_trim import ContextTrim, TrimStrategy


def example_basic_check_and_trim():
    """Check if messages fit; trim if they don't."""
    print("=== Basic: Check and Trim ===")

    # Simulate a long conversation
    messages = (
        [{"role": "system", "content": "You are a helpful assistant. Answer concisely."}]
        + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}: " + "detail " * 80}
            for i in range(40)
        ]
    )

    ct = ContextTrim(max_tokens=4096, reserved_tokens=512)

    print(f"Original: {len(messages)} messages, ~{ct.estimate(messages)} tokens")
    print(f"Fits:     {ct.fits(messages)}")

    if not ct.fits(messages):
        result = ct.trim(messages, strategy=TrimStrategy.HYBRID)
        print(result.summary())
        messages = result.messages

    print(f"Final:    {len(messages)} messages")
    print()


def example_strategies():
    """Compare all five trim strategies."""
    print("=== Strategy Comparison ===")

    messages = (
        [{"role": "system", "content": "System context."}]
        + [{"role": "user", "content": "detail " * 100} for _ in range(20)]
    )

    ct = ContextTrim(max_tokens=2000, reserved_tokens=200)

    for strategy in TrimStrategy:
        result = ct.trim(messages, strategy=strategy)
        print(f"  {strategy.value:20s}: {result.original_count}->{result.final_count} msgs | "
              f"{result.original_tokens}->{result.final_tokens} tok | "
              f"{'OK' if result.within_budget else 'OVER'}")
    print()


def example_ci_gate():
    """Demonstrate the CI gate."""
    print("=== CI Gate ===")

    short_messages = [{"role": "user", "content": "Hi, how are you?"}]
    long_messages = [{"role": "user", "content": "detail " * 5000}]

    ct = ContextTrim(max_tokens=4096, reserved_tokens=512)

    # Should pass
    try:
        ct.ci_gate(short_messages)
        print("Short messages: CI gate PASSED")
    except RuntimeError as e:
        print(f"Short messages: {e}")

    # Should fail
    try:
        ct.ci_gate(long_messages)
        print("Long messages: CI gate PASSED (unexpected)")
    except RuntimeError as e:
        print(f"Long messages: CI gate raised — {str(e)[:80]}")
    print()


def example_document_trim():
    """Trim a long document."""
    print("=== Document Trim ===")

    long_doc = "\n\n".join(
        f"Section {i}: " + "This section contains important information about the topic. " * 20
        for i in range(15)
    )

    ct = ContextTrim(max_tokens=1000, reserved_tokens=100)
    result = ct.trim_document(long_doc, strategy=TrimStrategy.HYBRID)
    print(result.summary())
    print(f"Text preview: {result.text[:100]}...")
    print()


def example_with_history(tmp_path_str: str = "/tmp"):
    """Store and query trim history."""
    import os
    db = os.path.join(tmp_path_str, "example_history.db")

    print("=== Trim History ===")

    ct = ContextTrim(max_tokens=2000, reserved_tokens=200, db_path=db)
    messages = [{"role": "user", "content": "detail " * 80} for _ in range(15)]

    for i in range(3):
        ct.trim(messages, pipeline_id="my-chatbot")

    from context_trim import TrimStore
    store = TrimStore(db_path=db)
    stats = store.stats("my-chatbot")
    print(f"Stats: {stats}")
    print()


if __name__ == "__main__":
    example_basic_check_and_trim()
    example_strategies()
    example_ci_gate()
    example_document_trim()
    print("All examples completed.")
