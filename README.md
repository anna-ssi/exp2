### Exploration-exploitation in humans

To start an environment with the dependencies do the following:
```
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

To run the code type the following command in the terminal:
```
python main.py -e experiments/eeg_train.json
```

To use wandb, first sign up and do the following in the terminal and paste your API key:
```
wandb login
```