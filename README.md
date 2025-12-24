### Requirements
You can install the required packages with the following command:
```bash
conda create -name ER_eval python==3.11.11
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
