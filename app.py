from flask import Flask, render_template, request
from simulation import run_simulation  # Your function from simulation.py

app = Flask(__name__)

# Home page with the form
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/simulate', methods=['POST'])
def simulate():
    agent_type = request.form['agent']
    rounds = int(request.form['rounds'])

    # Call your simulation function
    image_base64, metrics = run_simulation(agent_type, rounds)

    return render_template('result.html', image=image_base64, metrics=metrics, agent=agent_type)

if __name__ == '__main__':
    app.run(debug=True)
