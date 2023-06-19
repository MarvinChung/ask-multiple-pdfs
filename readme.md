#### Run code
1. Install requirements
```
pip3 install -r requirements.txt
```
2. Run the code
```
streamlit run app.py
```

#### Docker
1, Build docker
```
docker build -t ask-multiple-pdfs .
```
2. Run docker
```
docker run -it -p 8501:8501 ask-multiple-pdfs
```