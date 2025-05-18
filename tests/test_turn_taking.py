import types
import asyncio

import main

DialogicTurnCoordinator = main.DialogicTurnCoordinator
ConversationMemory = main.ConversationMemory
Message = main.Message
ModelProvider = main.ModelProvider
ModelRole = main.ModelRole
ModelConfig = main.ModelConfig


class DummyRegistry:
    def __init__(self):
        self.model1 = ModelConfig(
            api_name="m1",
            client=None,
            provider=ModelProvider.HUMAN,
            role=ModelRole.PRIMARY,
        )
        self.model2 = ModelConfig(
            api_name="m2",
            client=None,
            provider=ModelProvider.HUMAN,
            role=ModelRole.CRITIC,
        )
        self.models = {"model1": self.model1, "model2": self.model2}

    def get_model(self, name):
        return self.models[name]


class DummyGenerator:
    def __init__(self, responses):
        self.responses = responses
        self.count = 0

    async def generate_response(self, model_name, memory, prompt=None):
        resp = self.responses[self.count]
        self.count += 1
        return resp


def test_interrupt_stops_cycle():
    registry = DummyRegistry()
    gen = DummyGenerator(["hello", "stop [INTERRUPT]", "ignored"])
    mem = ConversationMemory(max_tokens=100)
    mem.add_message(Message(role="user", content="hi"))

    coordinator = DialogicTurnCoordinator(["model1", "model2"], registry, gen, max_cycles=3)
    asyncio.run(coordinator.execute(mem))

    msgs = mem.get_messages()
    assert len(msgs) == 3  # starter + two responses
    assert "INTERRUPT" in msgs[-1].content.text.upper()

