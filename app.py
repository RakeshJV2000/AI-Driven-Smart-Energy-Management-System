from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import openai
import os
import panel as pn  # GUI
from model import process_and_model_energy_data 
from twilio.rest import Client


client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)



# nlp = spacy.load("en_core_web_sm")


# def extract_keywords(text):
#     # Process the text
#     doc = nlp(text)
#     # Extract keywords (e.g., nouns for simplicity)
#     keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     return keywords

# def generate_questions(keywords):
#     # Simple template-based questions
#     questions = [f"What is the significance of {keyword}?" for keyword in keywords]
#     return questions


DATA = [
    {
        "prompt": "Energy Assistant, how can I reduce my electricity bill?",
        "response": "To reduce your electricity bill, consider turning off lights and appliances when not in use, using energy-efficient LED bulbs, and unplugging electronics when they're fully charged. You can also optimize your heating and cooling systems for energy savings."
    },
    {
        "prompt": "What are some energy-saving tips for my home appliances?",
        "response": "For energy-saving tips, try setting your thermostat to a comfortable yet efficient temperature, using cold water for laundry, and air-drying dishes instead of using the dishwasher's drying cycle. Additionally, consider upgrading to ENERGY STAR-rated appliances for long-term savings."
    },
    {
        "prompt": "Energy Assistant, can you provide some safety tips for using electrical devices?",
        "response": "Safety is important when using electrical devices. Always follow manufacturer instructions, avoid overloading outlets, and keep flammable materials away from heat sources. Be sure to unplug appliances before cleaning or servicing them, and regularly inspect cords and plugs for signs of wear or damage."
    },
    {
        "prompt": "What are the current trends in my energy consumption?",
        "response": "To check the current trends in your energy consumption, review your energy usage data from your smart meter or utility provider's online portal. Look for patterns in daily and weekly usage, and consider using energy monitoring devices to track specific appliances or areas of your home."
    },
    {
        "prompt": "Energy Assistant, how can I monitor and manage my energy usage more effectively?",
        "response": "To monitor and manage your energy usage effectively, consider installing a smart energy monitoring system that provides real-time data on your energy consumption. Set energy usage goals, track your progress over time, and adjust your habits accordingly to achieve savings."
    },
    {
        "prompt": "What are the benefits of using energy-efficient appliances?",
        "response": "Energy-efficient appliances can help you save money on your energy bills, reduce your carbon footprint, and contribute to a more sustainable environment. They often use less energy to perform the same tasks as standard appliances, resulting in lower operating costs and reduced greenhouse gas emissions."
    },
    {
        "prompt": "How can I improve the energy efficiency of my home?",
        "response": "To improve the energy efficiency of your home, consider adding insulation to reduce heat loss, sealing air leaks around windows and doors, and upgrading to ENERGY STAR-rated windows and appliances. You can also install programmable thermostats and LED lighting for additional savings."
    },
    {
        "prompt": "Energy Assistant, what are some common energy-wasting habits to avoid?",
        "response": "Common energy-wasting habits to avoid include leaving lights and appliances on when not in use, setting the thermostat too high or too low, and neglecting to perform routine maintenance on HVAC systems. By being mindful of your energy usage habits, you can reduce waste and save money."
    },
    {
        "prompt": "How can I make my home more energy-efficient during the winter months?",
        "response": "To make your home more energy-efficient during the winter, try sealing drafts around windows and doors, using a programmable thermostat to lower the temperature when you're away, and adding extra insulation to your attic and walls. You can also use a humidifier to add moisture to the air, which can make your home feel warmer at lower temperatures."
    },
    {
        "prompt": "What are some energy-saving tips for heating and cooling my home?",
        "response": "For heating and cooling savings, try setting your thermostat to a lower temperature in the winter and a higher temperature in the summer, using ceiling fans to circulate air, and closing curtains or blinds during the hottest part of the day. You can also schedule regular HVAC maintenance to ensure optimal performance and efficiency."
    },
    {
        "prompt": "Energy Assistant, how can I reduce phantom energy usage?",
        "response": "Phantom energy usage, also known as standby power, occurs when electronics and appliances continue to draw power even when turned off. To reduce phantom energy usage, unplug devices when not in use, use power strips to easily disconnect multiple devices, and choose ENERGY STAR-rated products that consume less standby power."
    },
    {
        "prompt": "How can I improve the energy efficiency of my lighting?",
        "response": "To improve the energy efficiency of your lighting, consider switching to LED bulbs, which use significantly less energy than incandescent or CFL bulbs and last much longer. You can also use dimmer switches and motion sensors to control lighting levels and reduce unnecessary usage."
    },
    {
        "prompt": "Energy Assistant, what are some energy-saving tips for cooking and kitchen appliances?",
        "response": "For cooking and kitchen appliances, try using energy-efficient appliances such as induction cooktops and convection ovens, which use less energy than traditional models. Additionally, avoid preheating the oven longer than necessary, use lids on pots and pans to retain heat, and only run the dishwasher when it's full."
    },
    {
        "prompt": "How can I make my laundry routine more energy-efficient?",
        "response": "To make your laundry routine more energy-efficient, try washing clothes in cold water whenever possible, using a high-efficiency washing machine, and air-drying clothes instead of using the dryer. You can also wash full loads of laundry to maximize energy and water usage."
    },
    {
        "prompt": "Energy Assistant, what are some energy-saving tips for water heating?",
        "response": "For water heating savings, try lowering the temperature on your water heater to 120 degrees Fahrenheit, insulating hot water pipes to reduce heat loss, and fixing leaks promptly to prevent wasted water and energy. You can also install low-flow showerheads and faucets to reduce water usage while still maintaining pressure."
    },
    {
        "prompt": "How can I save energy when using electronics and entertainment devices?",
        "response": "To save energy with electronics and entertainment devices, turn off devices when not in use, use sleep or power-saving modes whenever possible, and choose energy-efficient models with ENERGY STAR certification. You can also use power strips to easily disconnect multiple devices and reduce standby power consumption."
    }
]


DATA.extend([
    {
        "prompt": "What is a linear regression model?",
        "response": "A linear regression model predicts a dependent variable using a linear equation to the independent variables. It's commonly used for forecasting and finding out the relationship between variables."
    },
    {
        "prompt": "How do I handle missing data in my dataset?",
        "response": "Missing data can be handled in several ways, such as imputation, where you fill in the missing values based on other data, or by removing data points or features that have missing values, depending on the extent and nature of the missing data."
    },
    {
        "prompt": "Can you explain the concept of overfitting in machine learning?",
        "response": "Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means the model is too complex, capturing patterns that do not generalize to unseen data."
    },
    {
        "prompt": "What are the best practices for data visualization?",
        "response": "Best practices for data visualization include choosing the right type of chart for your data, keeping the design simple and uncluttered, using color effectively to highlight important information, and making sure the visualization is accessible and clear."
    },
    {
        "prompt": "How can I improve the accuracy of my predictive model?",
        "response": "Improving the accuracy of a predictive model can be done by using more or better quality data, trying different algorithms, tuning model parameters, and using techniques like cross-validation to better understand model performance."
    }
])

con = f"""
  You are a multifunctional assistant chatbot with expertise in energy savings and data science. 
  You provide tips on controlling energy use, maintaining safety, and understanding electrical appliances. 
  You inform users about energy usage, voltage requirements, potential issues, and how to address these concerns. 
  In the realm of data science, you offer explanations on data analysis techniques, statistical models, and best practices for data handling.
  Your responses are professional and focused, aimed at offering helpful guidance tailored to the specific queries about energy, power, 
  electrical topics, or data science questions.

  Example questions and answers in these domains are provided in this JSON data array: '''{DATA}''' with 'prompt' as key and 'response' as value.
"""

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


messages = [ {"role": "system", "content": con}]

def get_completion(model="gpt-3.5-turbo"):
    response = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages,
      temperature=0.7
  )
    return response.choices[0].message.content

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/get-response', methods=['POST'])
def get_response():
    # Get the user input from the request
    user_input = request.json['message']

    new_dict = {"role": "user", "content": user_input}
    messages.append(new_dict)
    response = get_completion()
    messages.append({"role": "assistant", "content": response})

    # Return the bot response as JSON
    return jsonify({'message': response})

@app.route('/process-csv', methods=['POST'])
def process_csv():
    if 'csv' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['csv']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        # Convert the uploaded CSV file to a DataFrame
        df = pd.read_csv(file)
        # Process the DataFrame with your ML model here
        results = process_and_model_energy_data(df) 
        return jsonify(result=results)
    else:
        return jsonify(error="Invalid file format"), 400

@app.route('/send-analysis', methods=['POST'])
def send_analysis():
    try:
        content = request.json
        mse = content.get('mse')
        mae = content.get('mae')

        # Creating the combined input that would have been sent to OpenAI
        combined_input = f"Mean Squared Error of the model is {mse} and mean absolute error is {mae}. Can you provide insights or suggestions based on these statistics for energy data?"
        next_input = "which algorithm should be used to reduce the error?"
        # Instead of sending to OpenAI, we just return this as a response
        return jsonify({'message': combined_input, 'message2': next_input})
    except Exception as e:
        app.logger.error(f"Failed to generate question: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/send-whatsapp-message', methods=['POST'])
def send_whatsapp_message():
    # Retrieve message from the incoming request's JSON
    content = request.get_json()  # get_json() parses the JSON input
    message_to_send = content['message']

    try:
        message = client.messages.create(
            body=message_to_send,
            from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
            to='whatsapp:+16317101230'       # Replace with the recipient's WhatsApp number
        )
        return jsonify({'status': 'Message sent', 'sid': message.sid})
    except Exception as e:
        return jsonify({'status': 'Failed to send message', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8080)
