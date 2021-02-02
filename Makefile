train:
	python graphsage/main.py

data/cora:
	git clone https://github.com/tkipf/pygcn.git
	mv pygcn/data/* data/
	rm -rf pygcn
