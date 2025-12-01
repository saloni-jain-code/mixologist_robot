# Bartending Robot 🤖🍹
COMSW4733 Final Course Project
Authors: Charles Chen, Saloni Jain, Tiffany Fu

[Milestone 1 Report](https://github.com/user-attachments/files/23061001/COMSW4733_Milestone_1_Report.pdf)

## Running the simulation
* Clone the repo
* In your terminal:
  * Create a virtual environment: `python -m venv robot`
  * Run `source robot/bin/activate`
  * Run `pip install -r requirements.txt`
* Create a `.env` file:
  * Add your Gemini API key in the following format: `GEMINI_API_KEY=`
* Run `python pick_up_and_rotate.py`
* You can specify a pour level of `low`, `medium`, or `high` by providing it as a CLI arg. If no args are provided, the robot will use a `medium` pour level.
  * Ex: `python pick_up_and_rotate.py high`
