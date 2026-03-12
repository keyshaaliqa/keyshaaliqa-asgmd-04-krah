from sklearn.model_selection import train_test_split
from data_ingest import load_data
from preprocess import create_preprocessor, save_preprocessor
from train import train_model, save_model
from evaluate import evaluate_model

def run_pipeline():

    df = load_data("Dataset/train.csv")

    y = df["Transported"]
    X = df.drop(columns=["Transported"])

    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall",
                "Spa", "VRDeck"]

    cat_cols = ["HomePlanet", "CryoSleep", "Cabin",
                "Destination", "VIP"]

    preprocessor = create_preprocessor(num_cols, cat_cols)

    X_processed = preprocessor.fit_transform(X)

    save_preprocessor(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    save_model(model)

    acc, report = evaluate_model(model, X_test, y_test)

    print("Accuracy:", acc)
    print(report)


if __name__ == "__main__":
    run_pipeline()