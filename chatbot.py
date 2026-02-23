from memory import Memory

class ChatBot:
    def __init__(self):
        self.memory = Memory()
        self.memory.load()

    def handle_greeting(self):
        return "Hello!"

    def set_user_name(self, name: str) -> str:
        name = name.strip().capitalize()
        if not name:
            return "Please tell me your name."

        self.memory.name = name
        self.memory.awaiting_name = False
        self.memory.save()
        return f"Nice to meet you, {name}."

    def handle_unknown(self):
        return "Tell me more."

    def reply(self, text: str) -> str:
        raw = text.strip()
        text = raw.lower()

        # Exit
        if text in ["bye", "exit", "quit"]:
            return "__exit__"

        # If we previously asked for name
        if self.memory.awaiting_name:
            return self.set_user_name(raw)

        # Greeting
        if text in ["hi", "hello"]:
            return self.handle_greeting()

        # Trigger name question
        if text in ["what's my name", "what is my name", "do you know my name"]:
            if not self.memory.name:
                self.memory.awaiting_name = True
                return "I don't know your name yet. What is it?"
            return f"You are {self.memory.name}."

        # User declares name
        if "my name is" in text:
            return self.set_user_name(raw)

        return self.handle_unknown()

