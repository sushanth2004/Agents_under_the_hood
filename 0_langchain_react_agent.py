from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Optional

load_dotenv()


@tool
def get_product_price(product: str) -> Optional[float]:
    """Returns the price of the product by looking up into the catalog"""

    product = product.lower().strip()

    print(f"Executing get_product_price(product: {product}) tool")

    catalog = {
        "laptop": 1250.99,
        "mobile": 650.77,
        "headphones": 350.00,
        "speakers": 450.65,
    }

    return catalog.get(product)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: gold, silver, bronze
    """

    print(
        f"Executing apply_discount(price: {price}, discount_tier: {discount_tier}) tool"
    )

    discount_percentages = {
        "gold": 25,
        "silver": 15,
        "bronze": 7,
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


llm = ChatOpenAI(
    model="gpt-5",
    temperature=0
)

tools = [get_product_price, apply_discount]

agent = create_agent(
    model=llm,
    tools=tools
)

messages = [
    SystemMessage(
        content=(
            "You are a helpful shopping assistant. "
            "You have access to product catalog tool and discount tool.\n\n"

            "Strict rules you must follow exactly:\n"

            "1. Never guess or assume any product price. "
            "You must call get_product_price first.\n"

            "2. Only call apply_discount after receiving the exact price "
            "from get_product_price.\n"

            "3. Never calculate discounts yourself. "
            "Always use apply_discount tool.\n"

            "4. If the user does not specify the discount tier, "
            "ask them which tier to use."
        )
    ),
    HumanMessage(
        content="what is the price of laptop after applying gold discount?"
    ),
]

result = agent.invoke({"messages": messages})

print(result["messages"][-1].content)