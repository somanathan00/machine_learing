from flask import Flask
from flask import render_template,request,jsonify
import pickle

app=Flask(__name__)
tokenizer=pickle.load(open('./models/cv.pkl','rb'))
model=pickle.load(open('./models/clf.pkl','rb'))

@app.route('/', )
def home():    
    return render_template("index.html",)

@app.route("/predict",methods=['POST'])
def predict():
    if request.method == "POST":
        email_text=request.form.get('email_body')
    tokenized_email=tokenizer.transform([email_text])
    predictions=model.predict(tokenized_email)
    predictions = "SPAM" if predictions==1 else "NOT SPAM"
    return render_template('index.html',predictions=predictions, text=email_text)

@app.route('/api/predict',methods=['POST'])
def api_predict():
    body=request.get_json(force=True)
    email=body.get('email_body')
    tokenized_email=tokenizer.transform([email])
    predictions=model.predict(tokenized_email)
    predictions = "SPAM" if predictions==1 else "NOT SPAM"
    return jsonify(predictions)

if __name__=="__main__":
    app.run(port=5000, debug=True)