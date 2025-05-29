from data_preprocessing import load_orders, orders_to_transactions, encode_transactions
from apriori_model import AprioriModel

ORDERS_PATH = '../data/orders.json'
RULES_PATH = './rules.csv'

def train():
    # Load & prepare data
    orders = load_orders(ORDERS_PATH)
    transactions = orders_to_transactions(orders)
    df_onehot = encode_transactions(transactions)

    # Train and save rules
    model = AprioriModel(min_support=0.05, min_confidence=0.08)
    rules = model.fit(df_onehot)
    model.save_rules(RULES_PATH)
    print(f"Rules saved to {RULES_PATH}")

if __name__ == '__main__':
    train()
