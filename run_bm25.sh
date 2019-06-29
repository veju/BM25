
export PYTHONPATH=$PWD/src
python3 -m ranking --model bm25 --docs text/corpus.full.txt --queries text/queries.all.txt