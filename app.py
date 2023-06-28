from flask import Flask , render_template , request
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pickle
app= Flask(__name__)
model= pickle.load(open('models/model.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')
@app.route('/submit',methods=["POST"])



def submit():
   if(request.method=="POST"):
      temp= float(request.form["temperature"])
      vacc= float(request.form["vacuum"])
      press=float(request.form["pressure"])
      humi= float(request.form["humidity"])
      
      g=pd.DataFrame({'Temp':temp,'Vacuum':vacc,'Pressure':press,'Humidity':humi},index=[0])
      
      
      poly = PolynomialFeatures( degree = 4, interaction_only= True)
      x_poly = poly.fit_transform(g)
      prediction=model.predict(x_poly)

      output= round(prediction[0],2)
      return render_template("index.html",prediction_text=output)

if __name__ == "__main__":
   app.run(debug=True)

