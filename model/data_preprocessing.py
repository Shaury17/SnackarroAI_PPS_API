import json
import pandas as pd

def load_orders(filepath: str):
    with open(filepath, 'r') as f:
        orders = json.load(f)
    return orders

def orders_to_transactions(orders):
    # Convert list of dicts to list of transactions (list of items)
    transactions = [order['items'] for order in orders]
    return transactions

def encode_transactions(transactions):
    """One-hot encode transactions for mlxtend"""
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df
