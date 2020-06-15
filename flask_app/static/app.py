from flask import Flask, render_template, request


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/blogin")
def blogin():
    return render_template("blogin.html")
@app.route("/bsign")
def bsign():
    return render_template("bsign.html")


if __name__ == "__main__":
    app.run(debug=True)
