all: build/v51.pdf

build/v51.pdf: build/invertertieren.pdf build/differentiator.pdf build/integrator.pdf v51.tex aufbau.tex auswertung.tex diskussion.tex durchfuehrung.tex fehlerrechnung.tex lit.bib theorie.tex ziel.tex | build
	lualatex  --output-directory=build v51.tex
	lualatex  --output-directory=build v51.tex
	biber build/v51.bcf
	lualatex  --output-directory=build v51.tex

build/invertertieren.pdf: invertieren.py invertieren1.txt  
	python invertieren.py

build/integrator.pdf: integrator.py integrator.txt
	python integrator.py

build/differentiator.pdf: differentiator.py differentiator.txt
	python differentiator.py 


build: 
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean all
