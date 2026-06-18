.PHONY: uml

uml:
	mkdir -p uml
	pyreverse -o png -p qSNOW -d uml interface/chip.py interface/models.py
