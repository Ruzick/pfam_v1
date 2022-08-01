FROM python:3.7

WORKDIR /pfam2
ADD pfam ./pfam
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r pfam/requirements.txt
ADD checkpoints ./checkpoints
ADD random_split ./random_split



CMD [ "python", "pfam/inference.py" ]