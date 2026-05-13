from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# ------Tool (Langchain @tool decorator)---------------


@tool
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


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers : gold, silver, bronze"""
    print(
        f"Excecuting apply_discount(price : {price}), discount_tier : {discount_tier}) tool"
    )

    discount_percentages = {"gold": 25, "silver": 15, "bronze": 7}

    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# ----------Agent Loop -------------


##manually adding langsmith tracing to function
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):

    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    # print(tools_dict)

    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question : {question}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"-----Iteration : {iteration}-----")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        ## if no tool calls , this is final answer
        if not tool_calls:
            print(f"Final Answer : {ai_message.content}")
            return ai_message.content

        ## process only one the First tool call - one tool call per iteration

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR : Max Iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello from Langchain Agent (.bind_tools!)")
    print()
    result = run_agent("what is the price of laptop after applying gold discount")
