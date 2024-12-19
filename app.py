from flask import Flask, request, render_template
from src.config import Config
from src.recommendation import Recommender

app = Flask(__name__)

recommender = Recommender()
recommender.load_interactions()

interaction_df = recommender.interaction_df
users = interaction_df["user_id"].unique()
categories = interaction_df["category"].unique()
occasions = interaction_df["rented for"].unique()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        category = request.form.get("category")
        occasion = request.form.get("occasion")
        top_n = int(request.form.get("top_n", Config.NUM_RECOMMENDATIONS))

        recommendations = recommender.recommend_items(user_id, occasion, category, top_n=top_n)
        rec_records = recommendations.to_dict(orient="records")
        return render_template("results.html", 
                               recommendations=rec_records,
                               user_id=user_id,
                               category=category,
                               occasion=occasion)
    else:
        return render_template("index.html", users=users, categories=categories, occasions=occasions)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

