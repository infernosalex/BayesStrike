from app import create_app

app = create_app(model_path="models/web_demo.json")

if __name__ == "__main__":
    app.run(debug=True)
