from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load and preprocess rules
df = pd.read_csv('model/rules.csv')

# Convert string frozensets to actual Python sets
df['antecedents'] = df['antecedents'].apply(lambda x: set(eval(x.replace("frozenset", ""))))
df['consequents'] = df['consequents'].apply(lambda x: set(eval(x.replace("frozenset", ""))))

@app.route('/', methods=['GET'])
def home():
    return '''Past purchases API'''

@app.route('/recommend/<string:txt>', methods=['GET'])
def recommend(txt):
    item = txt
    if not item:
        return jsonify({'error': 'Please provide an item'}), 400

    # Filter rules where the item is in the antecedents
    matching_rules = df[df['antecedents'].apply(lambda x: item in x)]

    if matching_rules.empty:
        return jsonify({'message': f'No recommendations found for {item}'}), 404

    # Sort rules by confidence (desc), then lift (desc)
    sorted_rules = matching_rules.sort_values(by=['confidence', 'lift'], ascending=False)

    # Extract top 2 unique recommended items
    seen_items = set()
    top_recommendations = []

    for _, row in sorted_rules.iterrows():
        for rec_item in row['consequents']:
            if rec_item not in seen_items:
                seen_items.add(rec_item)
                top_recommendations.append({
                    'recommended_items': [rec_item],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                })
            if len(top_recommendations) >= 2:
                break
        if len(top_recommendations) >= 2:
            break

    return jsonify({
        'input_item': item,
        'recommendations': top_recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
