from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    """Generates the greetings for user"""
    return f"Hell {name} , Welcome to Mumbai , the city of Dreams"

result = get_greeting.invoke({"name" : "Rajvardhan"})
print(result)
print('='*50)
print(get_greeting.args)
print(get_greeting.description)
print(get_greeting.name)