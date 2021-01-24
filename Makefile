train:
	python model/main.py

data:
	git clone https://github.com/tkipf/pygcn.git
	mv pygcn/data .
	rm -rf pygcn
