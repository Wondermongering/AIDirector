# import main after dummy modules have been set in conftest
import main

Message = main.Message
MessageContent = main.MessageContent
ConversationMemory = main.ConversationMemory
TokenLimitStrategy = main.TokenLimitStrategy


def create_message(text):
    return Message(role="user", content=text)


def test_truncate_oldest_strategy():
    mem = ConversationMemory(max_tokens=10, strategy=TokenLimitStrategy.TRUNCATE_OLDEST)
    mem.add_message(create_message("one two three four five six seven eight"))  # 8 tokens
    mem.add_message(create_message("nine ten eleven twelve thirteen fourteen"))  # 6 tokens
    # After exceeding limit, oldest message should be removed
    assert len(mem.get_messages()) == 1
    assert mem.get_messages()[0].content.text.startswith("nine")


def test_summarize_strategy_invoked(monkeypatch):
    called = {}

    def fake_summarize(self):
        called['summarize'] = True
        # mimic original behaviour
        ConversationMemory._truncate_oldest(self)

    mem = ConversationMemory(max_tokens=10, strategy=TokenLimitStrategy.SUMMARIZE)
    monkeypatch.setattr(ConversationMemory, "_summarize_conversation", fake_summarize)

    mem.add_message(create_message("one two three four five six seven eight"))
    mem.add_message(create_message("nine ten eleven twelve thirteen fourteen"))

    assert called.get('summarize', False)


def test_summarize_conversation_replaces_messages(monkeypatch):
    mem = ConversationMemory(max_tokens=10, strategy=TokenLimitStrategy.SUMMARIZE)

    monkeypatch.setattr(
        ConversationMemory,
        "_get_summary_from_model",
        lambda self, msgs: "summary",
    )

    for i in range(6):
        mem.add_message(create_message(f"{i} {i+1}"))

    msgs = mem.get_messages()
    assert msgs[0].content.text == "summary"
    assert len(msgs) == 4
    assert mem.total_tokens == 7


def test_sliding_window_strategy_invoked(monkeypatch):
    called = {}

    def fake_window(self):
        called['window'] = True
        ConversationMemory._truncate_oldest(self)

    mem = ConversationMemory(max_tokens=10, strategy=TokenLimitStrategy.SLIDING_WINDOW)
    monkeypatch.setattr(ConversationMemory, "_apply_sliding_window", fake_window)

    mem.add_message(create_message("one two three four five six seven eight"))
    mem.add_message(create_message("nine ten eleven twelve thirteen fourteen"))

    assert called.get('window', False)


def test_sliding_window_discards_old_tokens():
    mem = ConversationMemory(max_tokens=10, strategy=TokenLimitStrategy.SLIDING_WINDOW)

    mem.add_message(create_message("one two three four"))
    mem.add_message(create_message("five six seven eight"))
    mem.add_message(create_message("nine ten eleven"))

    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content.text.startswith("five")
    assert mem.total_tokens == 7

