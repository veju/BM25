
export PYTHONPATH=$PWD/src
python3 -m ranking --model tfidf --docs text/corpus.full.txt --queries text/queries.all.txt
