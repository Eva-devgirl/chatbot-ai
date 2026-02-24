import re
from unittest.mock import seal

from ml_intent import IntentModel
from memory import Memory

class ChatBot:
    def __init__(self):
        self.intent_model = IntentModel()
        self.memory = Memory.load()
        self.bot_name = "SimpleBot"
        self.min_confidence = 0.3 # threshold for unknown

    def reply(self, text: str) -> str:
        raw = text.strip()
        low = raw.lower()

        # 1) If we are waiting for the user to provide a name
        if self.memory.awaiting_name:
            return self._handle_name_answer(raw)

        # 2) Predict intent with ML
        intent, conf = self.intent_model.predict(low)
        print("DEBUG:", intent, conf)

        #3) Confidence gate (avoid random wrong intents)
        if conf < self.min_confidence and intent != "greeting":
            return self._handle_unknown()

        # 4) Route to handler
        if intent == "exit":
            return "__exit__"

        if intent == "greeting":
            return self._handle_greeting()

        if intent == "ask_user_name":
            return self._handle_ask_user_name()

        if intent == "set_user_name":
            return self._handle_set_user_name(raw)

        return self._handle_unknown()

    # Handlers


    def _handle_greeting(self) -> str:
        return "Hello!"

    def _handle_ask_user_name(self) -> str:
        if not self.memory.name:
            print("DEBUG name in memory = ", self.memory.name)
            self.memory.awaiting_name = True
            self.memory.save()
            return "I don't know who you are yet."

        return f"You are {self.memory.name}."

    def _handle_name_answer(self, raw: str) -> str:
        name = raw.strip().capitalize()
        if not name:
            return "Please tell me your name."

        self.memory.name = name
        self.memory.awaiting_name = False
        self.memory.save()
        return f"Nice to meet you, {self.memory.name}."

    def _handle_set_user_name(self, raw: str) -> str:
        """
        Extract a name from phrases like:
        - "my name is Eva"
        - "I'm Eva"
        - "call me Eva"
        If extraction fails, we ask again.
        """
        name = self._extract_name(raw)
        if not name:
            self.memory.awaiting_name = True
            self.memory.save()
            return "I didn't catch your name. Please tell me your name."

        self.memory.name = name
        self.memory.awaiting_name = False
        self.memory.save()
        return f"Nice to meet you, {self.memory.name}."

    def _handle_unknown(self):
        return "Tell me more."

    # Helpers

    def _extract_name(selfself, raw: str) -> str:
        s = raw.strip()

        patterns = [
            r"my name is\s+(.+)$",
            r"i am\s+(.+)$",
            r"i'm\s+(.+)$",
            r"call me\s+(.+)$",
            r"you can call me\s+(.+)$",
        ]

        low = s.lower()
        for p in patterns:
            m = re.search(p, low)
            if m:
                # take the matched group from the original string
                start = m.start(1)
                name_part = s[start:].strip()
                # keep only first token as a simple name (optional)
                first = name_part.split()[0]
                return first.capitalize()

        return None






