import pandas as pd
import re

def clean_ingredients(text):
    text = str(text).lower()
    stop_patterns = r'\d+|cup|tablespoon|tsp|tbsp|teaspoon|gram|gms|kg|finely|chopped|sliced|peeled|washed|grated|to taste'
    text = re.sub(stop_patterns, '', text)
    text = re.sub(r'[^a-zA-Z\s,]', '', text)
    return " ".join(text.split())

def build_model():
    print("Loading dataset...")
    df = pd.read_csv('IndianFoodDatasetCSV.csv')
    
    print("Preprocessing ingredients...")
    # Fill NaN to avoid 'nan' strings
    df['Ingredients'] = df['Ingredients'].fillna('')
    df['Clean_Ingredients'] = df['Ingredients'].apply(clean_ingredients)
    
    # Save the preprocessed data for faster loading in app.py
    print("Saving preprocessed dataset...")
    df.to_csv('preprocessed_recipes.csv', index=False)
    print("Done! You can now use preprocessed_recipes.csv in app.py for faster startup.")

if __name__ == "__main__":
    build_model()
