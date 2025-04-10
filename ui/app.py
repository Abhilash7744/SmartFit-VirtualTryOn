from flask import Flask, render_template, request
import os
from test import try_on

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        person = request.files["person"]
        garment = request.files["garment"]
        person_path = f"static/uploads/{person.filename}"
        garment_path = f"static/uploads/{garment.filename}"
        person.save(person_path)
        garment.save(garment_path)
        output_img = try_on(person_path, garment_path)
        output_img.save("static/result.jpg")
        return render_template("index.html", result="static/result.jpg")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
