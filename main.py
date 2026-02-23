from chatbot import ChatBot

def main():
    bot = ChatBot()

    print ("Simple AI ChatBot (type 'exit' to quit)")

    while True:
        user_input = input("You: ")
        response = bot.reply(user_input)

        if response == "__exit__":
            print("Bot: Goodbye!")
            break
        print ("Bot:", response)

if __name__ == "__main__":
   main()
