import pandas as pd
import random
from datetime import datetime, timedelta

def generate_bank_statement():
    # Setup
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    current_date = start_date
    
    data = []
    balance = 50000.0 # Opening Balance
    
    # Categories & Typical Amounts
    salary = 162133.0
    rent = 25000.0
    sip_investments = [
        ("ACH-ZERODHA-BROKING", 15000), 
        ("ACH-GROWW-SIP", 5000), 
        ("UPI-INDMONEY-US-STOCKS", 5000)
    ]
    utilities = [("BILL-ELECTRICITY", 1500, 3000), ("BILL-INTERNET", 999, 999), ("UPI-JIO-RECHARGE", 666, 666)]
    food = ["UPI-SWIGGY", "UPI-ZOMATO", "POS-STARBUCKS", "POS-MCDONALDS", "UPI-BLINKIT"]
    shopping = ["UPI-AMAZON", "UPI-FLIPKART", "POS-H&M", "POS-ZARA"]
    
    while current_date <= end_date:
        
        # 1. Salary (1st of month)
        if current_date.day == 1:
            balance += salary
            data.append({
                "Txn Date": current_date.strftime("%d-%m-%Y"),
                "Description": "ACH-SALARY-NEO-HORIZON TECH",
                "Cheque No": " ",
                "Debit": 0,
                "Credit": salary,
                "Balance": round(balance, 2)
            })
            
            # 2. Rent (2nd of month)
            current_date += timedelta(hours=2) # Same day slightly later or next day
            balance -= rent
            data.append({
                "Txn Date": current_date.strftime("%d-%m-%Y"),
                "Description": "UPI-RENT-LANDLORD-SHARMA",
                "Cheque No": " ",
                "Debit": rent,
                "Credit": 0,
                "Balance": round(balance, 2)
            })
            
            # 3. SIPs (5th of month) - Let's verify date logic later, putting them on 5th
        
        if current_date.day == 5:
            for desc, amt in sip_investments:
                balance -= amt
                data.append({
                    "Txn Date": current_date.strftime("%d-%m-%Y"),
                    "Description": desc,
                    "Cheque No": " ",
                    "Debit": amt,
                    "Credit": 0,
                    "Balance": round(balance, 2)
                })

        # Random Expenses (Every few days)
        if random.random() < 0.4: # 40% chance of distinct transaction each day
            cat = random.choice(["Food", "Shopping", "Utility"])
            if cat == "Utility" and current_date.day == 10: # Utilities around 10th
                desc_tpl = random.choice(utilities)
                amt = random.randint(desc_tpl[1], desc_tpl[2]) if len(desc_tpl) > 2 else desc_tpl[1]
                desc = desc_tpl[0]
            elif cat == "Food":
                desc = random.choice(food) + "-" + str(random.randint(100, 999))
                amt = random.randint(200, 1500)
            else: # Shopping
                desc = random.choice(shopping)
                amt = random.randint(500, 5000)
            
            # Record Expense
            if cat != "Utility" or current_date.day == 10:
                balance -= amt
                data.append({
                    "Txn Date": current_date.strftime("%d-%m-%Y"),
                    "Description": desc,
                    "Cheque No": " ",
                    "Debit": amt,
                    "Credit": 0,
                    "Balance": round(balance, 2)
                })

        current_date += timedelta(days=1)

    # Save
    df = pd.DataFrame(data)
    # Add metadata strings at the top to simulate real bank CSV
    with open("Rohan_Bank_Statement.csv", "w") as f:
        f.write("HDFC BANK STATEMENT,,,,,\n")
        f.write("Account Name: Rohan Gupta,,,,,\n")
        f.write("Address: 123 Cyber Hub Gurgaon,,,,,\n")
        f.write(",,,,,\n") # Empty line
        
    df.to_csv("Rohan_Bank_Statement.csv", mode='a', index=False)
    print(f"Generated {len(df)} transactions.")

if __name__ == "__main__":
    generate_bank_statement()
