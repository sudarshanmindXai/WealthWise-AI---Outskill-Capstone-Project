
import os
from dotenv import load_dotenv # Import this
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# 1. Load the Environment Variables
load_dotenv() # This reads the .env file

# 2. Get the Key safely
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Please check your .env file.")

# 3. Setup the LLM
llm = OpenAI(api_token=api_key, model="gpt-4o")


# 2. Load the Data (The Memory)
df = pd.read_csv("Rohan_Bank_Statement.csv")

# 3. Clean the Data (The Pre-processing)
# PandasAI is smart, but date formats in India (DD-MM-YYYY) confuse US-centric models.
# Let's force convert it so the AI doesn't hallucinate dates.
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# 4. Initialize the Smart Agent
# "config" helps us debug if the AI writes bad code
agent = SmartDataframe(
    df, 
    config={"llm": llm, "verbose": True}
)

# 5. The "Rohan" Context Injection
# We don't just ask "Sum descriptions". We give the AI a 'Lens' to view the data.
system_instruction = """
You are analyzing a bank statement for a user named Rohan.
Column 'Description' contains the merchant name.
Column 'Withdrawal' is money spent.
Column 'Deposit' is income.

Categorization Rules:
- 'Swiggy', 'Zomato', 'Starbucks', 'McDonalds' -> Food & Dining
- 'Zerodha', 'Groww', 'IndMoney' -> Investments
- 'Uber', 'Ola', 'Rapido' -> Travel
- 'Cred', 'Credit Card' -> Debt Payments
- 'LIC', 'Acko', 'Policy' -> Insurance
- 'Rent', 'Landlord' -> Rent
"""

def ask_agent(question):
    print(f"\nUser asks: {question}")
    # We combine the system instruction with the user question
    full_prompt = f"{system_instruction} \n Question: {question}"
    response = agent.chat(full_prompt)
    print(f"AI Answer: {response}")
    return response

# --- TEST ZONE ---
if __name__ == "__main__":
    # Test 1: Simple Retrieval
    ask_agent("What is the total salary credited?")

    # Test 2: Categorization (The Hard Part)
    ask_agent("Calculate the total spending on 'Food & Dining'.")
    
    # Test 3: The "Trap" Question (For the HRA Logic)
    ask_agent("Is there any transaction labeled as 'Rent' or 'Landlord'? Return Yes or No.")