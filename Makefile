# Test runner shortcuts (uv + the active virtual environment).
#
# Usage:
#    make test                     # run the whole unittest suite (discovery)
#    make test T=model.unittests.utilities.test_TimeIt.py
#    make test T=...test_TimeIt.Test_TimeIt   # a single case

UV := uv run --active

.PHONY: test

test:
ifdef T
    $(UV) python -m unittest $(T)
else
	$(UV) python -m unittest discover model
	$(UV) python -m unittest discover webscrapers
endif