# implmenting ReAct agent loop using raw ollama SDK

from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# ------Defining custom tools---------------

# Difference 1 : adding langsmith tracing to tool functions
# as we are not using tool decorator


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Returns the price of the product by looking up into the catelog"""
    print(f"Excecuting get_product_price(product : {product}) tool")
    catelog = {
        "laptop": 1250.99,
        "mobile": 650.77,
        "headphones": 350.00,
        "speakers": 450.65,
    }

    return catelog.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers : gold, silver, bronze"""
    print(
        f"Excecuting apply_discount(price : {price}), discount_tier : {discount_tier}) tool"
    )

    discount_percentages = {"gold": 25, "silver": 15, "bronze": 7}

    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# Difference 2
# converting the python into tools which llm can digest
# Without @tool, we must manually define json schema for each function.
# This is exactly what langchain's @tool decorator generates automatically using
# function's type hints and docstring

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of the item in catelog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name , eg: Laptop, headphone, etc",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers : gold, silver, bronze",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier bronze, silver or gold",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# ----------Agent Loop -------------


##manually adding langsmith tracing to function
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    # changed
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # print(tools_dict)

    print(f"Question : {question}")
    print("=" * 50)

    # Difference 4 : changed - as we cannot use Humman msg or system msg (langchain) here
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant"
                "you have access to product catelog tool and discount tool \n\n"
                "Strict rules you must follow these exactly : \n"
                "1. Never guess or assume any product price."
                "you must call get_product_price first to get the real price. \n"
                "2. Only call apply_discount after after you have "
                "recieved a price from get_product_price tool. Pass the exact price "
                "returned from get_product_price - do not pass a made up number.\n"
                "3. Never calculate discounts yourself using math."
                "Always use apply_discount tool.\n"
                "4. If the user does not specify the discount tier,"
                "ask them which tier to use - do not assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"-----Iteration : {iteration}-----")

        # Difference 5 : using ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        ## if no tool calls , this is final answer
        if not tool_calls:
            print(f"Final Answer : {ai_message.content}")
            return ai_message.content

        ## process only one the First tool call - one tool call per iteration

        tool_call = tool_calls[0]
        # Difference 6 : Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f"[Tool result] {observation}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

    print("ERROR : Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from Langchain Agent (raw function calling)")
    print()
    result = run_agent("what is the price of speakers after applying bronze discount")
