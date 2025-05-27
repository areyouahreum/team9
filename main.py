from recommender.dataset_loader import load_dataset
from recommender.knn_algorithm import get_user_vector, knn_algorithm, strict_matches
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def print_recipe_table(df):
    h_recipe = "recipe".center(30)
    h_match = "match".center(8)
    print(f"{h_recipe}{h_match}")
    print("-" * 38)

    for _, i in df.iterrows():
        recipe = str(i['recipe'].ljust(30))
        match  = f"{i['match']:.4f}".rjust(8)
        print(f"{recipe}{match}")

def recipe_choice(df):
    print("\nSelect a recipe: ")

    for i, j in enumerate(df.itertuples(index=False), 1):
        print(f"{i}. {j.recipe}")
    
    while True:
        try:
            choice = int(input("Enter recipe number: "))
            if 1 <= choice <= len(df):
                select = df.iloc[choice -1]
                print(f"\nYou selected: {select['recipe']}")
                instruction = select.get("instruction", "").strip()
                if instruction:
                    print("\nInstructions:\n" + instruction)
                else:
                    print("\n(Invalid instruction.)") 
                break
            else:
                print(f"Please enter a number between 1 and {len(df)}.")
        except ValueError:
            print("Invalid input. Please enter a number")

def main():
    print("Welcome to BakeHelp")
    df, ingredient_cols = load_dataset()
    X = df[ingredient_cols]

    user_input = input("Enter your ingredients: ").lower().strip().split(',')
    user_input = [i.strip() for i in user_input]
    user_vector = get_user_vector(ingredient_cols, user_input)
    print("\n=== Top Recipe Matches ===\n")
    top_match = knn_algorithm(df, X, user_vector)
    print_recipe_table(top_match)
    recipe_choice(top_match)

    strict = strict_matches(df, ingredient_cols, user_input)
    if not strict.empty:
        print("\nStrict Matches (You have all ingredients): ")
        print(strict[['recipe']].to_string(index=False))
    else:
        print("\nNo strict matches found.")

if __name__ == "__main__":
    main()
