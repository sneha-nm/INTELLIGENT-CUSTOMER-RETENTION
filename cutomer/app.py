from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	# Load the model and feature scaler
	model = joblib.load('churn.pkl')
	scaler = joblib.load('scaler.pkl')

	# Get user inputs from the form
	inputs = pd.DataFrame({
		'Account Length': [request.form['account_length']],
		'International Plan': [request.form['intl_plan']],
		'Voice Mail Plan': [request.form['vm_plan']],
		'Number vmail messages': [request.form['num_vmail']],
		'Total Day Minutes': [request.form['day_mins']],
		'Total Day Calls': [request.form['day_calls']],
		'Total Day Charge': [request.form['day_charge']],
		'Total Eve Minutes': [request.form['eve_mins']],
		'Total Eve Calls': [request.form['eve_calls']],
		'Total Eve Charge': [request.form['eve_charge']],
		'Total Night Minutes': [request.form['night_mins']],
		'Total Night Calls': [request.form['night_calls']],
		'Total Night Charge': [request.form['night_charge']],
		'Total Intl Minutes': [request.form['intl_mins']],
		'Total Intl Calls': [request.form['intl_calls']],
		'Total Intl Charge': [request.form['intl_charge']],
		'Customer Service Calls': [request.form['custserv_calls']]
	})

	# Scale the inputs using the feature scaler
	inputs_scaled = scaler.transform(inputs)

	# Make the prediction using the model
	prediction = model.predict(inputs_scaled)[0]

	# Render the appropriate template based on the prediction
	if prediction == 0:
		return render_template('predno.html')
	else:
		return render_template('predyes.html')

if __name__ == '__main__':
	app.run(debug=True)
