TEXT CLASSIFICATION PROJECT: NAIVE BAYES vs LLM

1. Install Python 3.10+ and open this folder in VS Code.
2. In PowerShell terminal run:
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
3. Optional: set your Claude API key in PowerShell:
   $env:ANTHROPIC_API_KEY="your-real-api-key"
   Do not share or upload your API key.
4. Run:
   python main.py
5. Enter a review. The program prints predictions and saves each input to
   classification_results.csv. It updates sentiment_comparison.png after each input.
6. Type quit to stop.

If no API key is configured, Naive Bayes still works. The LLM result will be saved
as unavailable. The graph compares the counts of predictions made by each model,
not accuracy. To compare accuracy, you need a labeled test dataset.
