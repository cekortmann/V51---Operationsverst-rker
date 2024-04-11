all: build/v51.pdf

build/v51.pdf: v51.tex aufbau.tex auswertung.tex diskussion.tex durchfuehrung.tex fehlerrechnung.tex lit.bib theorie.tex ziel.tex | build
	lualatex  --output-directory=build v51.tex
	lualatex  --output-directory=build v51.tex
	biber build/v51.bcf
	lualatex  --output-directory=build v51.tex


build: 
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean all
