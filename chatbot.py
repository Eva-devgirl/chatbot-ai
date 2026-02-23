from memory import Memory

class ChatBot:
    def __init__(self):
        self.memory = Memory()

    def handle_greeting(self):
        return "Hello!"

    def handle_set_name(self, text):
        name = text.strip().capitalize()
        self.memory.name = name
        self.memory.awaiting_name = False
        return f"Nice to meet you, {name}."

    def handle_unknown(self):
        return "Tell me more."

    def reply(self, text:str) -> str:
        text = text.lower().strip()

        # Exit
        if text in ["bye", "exit"]:
            return "__exit__"

        # If we previously asked for name
        if self.memory.awaiting_name:
            return self.handle_set(text)

        # Greeting
        if text in ["hi", "hello"]:
            return self.handle_greeting()

        # Ask name
        if text in ["what is my name", "what's my name", "who am i"]:
            if self.memory.name:
                return f"You are are {self.memory.name}."
            return "I don't know your name yet"

        # Trigger name question
        if text in ["what's my name", "do you know my name"]:
            if not self.memory.name:
                self.memory.awaiting_name = True
                return "I dont know your name. What is it?"
            return  f"you are {self.memory.name}."

        return self.handle_unknown()

