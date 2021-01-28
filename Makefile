train:
	python model/main.py

data/cora:
	git clone https://github.com/tkipf/pygcn.git
	mv pygcn/data .
	rm -rf pygcn
