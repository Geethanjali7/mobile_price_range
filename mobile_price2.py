import pickle
from flask import Flask,render_template,request


application=Flask(__name__)
try:
    with open('mp.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        scaler = model_data['scaler']

except (IOError, EOFError) as e:
    print("Error loading the pickled file:", e)
@application.route('/')
def fun():
    return render_template('mobile_price3.html')

def fun():
    return render_template('mobile_price3.html')
@application.route('/predict', methods=['POST'])
def predict():
    data=request.form
    features=[[int(data['battery_power']),int(data['blue']),float(data['clock_speed']),int(data['dual_sim']),int(data['fc']),
              int(data['four_g']),int(data['int_memory']),float(data['m_dep']),int(data['mobile_wt']),
              int(data['n_cores']),int(data['pc']),int(data['px_height']),int(data['px_width']),
              int(data['ram']),int(data['sc_h']),int(data['sc_w']),int(data['talk_time']),
              int(data['three_g']),int(data['touch_screen']),int(data['wifi'])
             ]]
    scaled_features=scaler.transform(features)
    prediction = model.predict(scaled_features)
    if prediction==0:
        output="cheap"
    elif prediction==1:
        output="affordable"
    elif prediction==2:
        output="expensive"
    else:
        output="very expensive"
    return render_template('mobile_price3.html',prediction_text="the price is {}".format(output))
if __name__ == "__main__":
    application.run(debug=True)
