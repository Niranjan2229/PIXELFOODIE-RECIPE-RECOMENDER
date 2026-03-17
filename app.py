import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import re

app = Flask(__name__)

# Try to load preprocessed data if it exists, otherwise load original
CSV_FILENAME = 'IndianFoodDatasetCSV.csv'
PREPROCESSED_FILENAME = 'preprocessed_recipes.csv'

if os.path.exists(PREPROCESSED_FILENAME):
    print(f"Loading preprocessed data from {PREPROCESSED_FILENAME}...")
    df = pd.read_csv(PREPROCESSED_FILENAME)
else:
    print(f"Loading original data from {CSV_FILENAME}...")
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        
        def clean_ingredients(text):
            text = str(text).lower()
            stop_patterns = r'\d+|cup|tablespoon|tsp|tbsp|teaspoon|gram|gms|kg|finely|chopped|sliced|peeled|washed|grated|to taste'
            text = re.sub(stop_patterns, '', text)
            text = re.sub(r'[^a-zA-Z\s,]', '', text)
            return " ".join(text.split())

        df['Clean_Ingredients'] = df['Ingredients'].fillna('').apply(clean_ingredients)
    else:
        print(f"Error: {CSV_FILENAME} not found!")
        df = pd.DataFrame(columns=['RecipeName', 'Ingredients', 'Instructions', 'Clean_Ingredients'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_input = data.get('ingredients', '').lower()
    
    # Split by comma and clean up
    user_ingredients = list(set([x.strip() for x in raw_input.split(',') if x.strip()]))
    
    if not user_ingredients: 
        return jsonify([])

    from typing import List, Dict, Any
    results: List[Dict[str, Any]] = []
    seen_recipes = set()

    # Iterate over the dataframe to find matches
    # Using itertuples for faster iteration than iterrows
    for row in df.itertuples():
        recipe_name = getattr(row, 'RecipeName')
        if recipe_name in seen_recipes: 
            continue
            
        clean_ingreds = str(getattr(row, 'Clean_Ingredients')).lower()
        if not clean_ingreds or clean_ingreds == 'nan':
            continue
            
        recipe_items_list = clean_ingreds.split()
        recipe_items_count = len(recipe_items_list)
        
        if recipe_items_count == 0: 
            continue

        # Count how many user ingredients are in the recipe ingredients
        matches = sum(1 for item in user_ingredients if item in clean_ingreds)
        user_coverage = matches / len(user_ingredients)  # type: ignore
        
        if user_coverage >= 0.3: # Lowered threshold slightly to be more helpful
            recipe_coverage = matches / recipe_items_count
            
            # Scoring logic
            final_score = (user_coverage * 0.8) + (recipe_coverage * 0.2)
            display_percent = min(round(final_score * 100), 100)
            
            results.append({
                "recipe_name": recipe_name,
                "ingredients": getattr(row, 'Ingredients'),
                "instructions": getattr(row, 'Instructions'),
                "match_val": final_score,
                "match": f"{display_percent}%"
            })
            seen_recipes.add(recipe_name)

    # Sort results by match score
    sorted_results = sorted(results, key=lambda x: x['match_val'], reverse=True)
    final_results = sorted_results[:20]  # type: ignore
    
    return jsonify(final_results)

if __name__ == '__main__':
    app.run(debug=True)