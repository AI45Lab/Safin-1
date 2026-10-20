PYTHON ?= python

.PHONY: test audit whl clean-build
test:
	$(PYTHON) -m pytest tests/march -q

audit:
	$(PYTHON) scripts/march/check_public_release.py .
	$(PYTHON) scripts/march/check_vendor_manifest.py --strict-provenance
	$(PYTHON) scripts/march/check_publication.py --allow-draft

clean-build:
	rm -rf build

whl: clean-build
	$(PYTHON) -m build
