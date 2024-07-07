# ===== VARIABLES =====
# include the configuration file
include Makefile.config

# Python version file
PYTHON_VERSION_FILE = .python-version

# Python interpreters
PYENV_PYTHON = $(shell pyenv root)/versions/$(PYTHON_VERSION)/bin/python
PYTHON = $(VENV_DIR)/bin/python

# requirements
COMPILED_REQUIREMENTS_FILE = requirements.txt
ifeq ($(DEV), true)
	REQUIREMENTS_FILES = requirements.in requirements-dev.in
else
	REQUIREMENTS_FILES = requirements.in
endif

# virtual environment
VENV_DIR = venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

# directories
DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/raw
COMPLEMENTARY_DATA_DIR = $(DATA_DIR)/complementary
MODELS_DIR = models
NOTEBOOKS_DIR = notebooks
SUBMISSIONS_DIR = submissions
TESTS_DIR = tests
DIRECTORIES = $(DATA_DIR) \
							$(MODELS_DIR) \
							$(NOTEBOOKS_DIR) \
							$(SRC_DIR) \
							$(SUBMISSIONS_DIR) \
							$(TESTS_DIR) \
							$(RAW_DATA_DIR) \
							$(COMPLEMENTARY_DATA_DIR) \
							$(PACKAGE_DIR)

# raw data zip file
RAW_DATA_ZIP = $(RAW_DATA_DIR)/$(COMPETITION_NAME).zip


# ===== INIT & CLEAR =====
# initialise the project
.PHONY : init
init :
	$(MAKE) install-requirements
	$(MAKE) install-pre-commit-hooks
	$(MAKE) create-directories
	$(MAKE) rename-package
ifeq ($(LOAD_DATA_ON_INIT), true)
	$(MAKE) load-data
endif

# create directories structure for the project
.PHONY : create-directories
create-directories : 
	for dir in $(DIRECTORIES); do \
		mkdir -p $$dir; \
	done

# rename the python package to the competition name
.PHONY : rename-package
rename-package :
	mv src/competition_name src/$(subst -,_,$(COMPETITION_NAME))

# reset the package name to "competition_name"
.PHONY : reset-package-name
reset-package-name :
	mv src/* src/competition_name

# reset the project but keep user created code
.PHONY : clear
clear :
	rm -rf $(PYTHON_VERSION_FILE) $(COMPILED_REQUIREMENTS_FILE) $(VENV_DIR) $(RAW_DATA_DIR) .git/hooks
	$(MAKE) reset-package-name


# ===== LINTING & FORMATTING =====
# lint and format
.PHONY : lint-format
lint-format : | $(VENV_ACTIVATE)
	$(MAKE) lint
	$(MAKE) format

# lint
.PHONY : lint
lint : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && ruff check

# format
.PHONY : format
format : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && ruff format


# ===== TESTING =====
# run tests
.PHONY : test
test : | $(VENV_ACTIVATE)
	$(PYTHON) -m unittest


## ===== JUPYTER =====
# start a Jupyter server
.PHONY : jupyter
jupyter : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && ipython kernel install --user --name=$(VENV_DIR) && env PYTHONPATH=`pwd` jupyter notebook


# ===== DATA =====
# download the raw data, unzip it and delete the archive
.PHONY : load-data
load-data : unzip-raw-data

# remove data directory
.PHONY : clear-data
clear-data :
	rm -rf $(DATA_DIR)

# download the raw data zip using the Kaggle API
$(RAW_DATA_ZIP) : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && kaggle competitions download $(COMPETITION_NAME) -p $(RAW_DATA_DIR)

# unzip the raw data archive and delete it
.PHONY : unzip-raw-data
unzip-raw-data : | $(RAW_DATA_ZIP)
	unzip $| -d $(RAW_DATA_DIR)
	rm $|


# ===== PYTHON =====
# install requirements
.PHONY : install-requirements
install-requirements : $(COMPILED_REQUIREMENTS_FILE)
	. $(VENV_ACTIVATE) && pip-sync

# compile requirements
$(COMPILED_REQUIREMENTS_FILE) : $(REQUIREMENTS_FILES) | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && pip-compile --output-file=$@ --strip-extras $^
	
# create virtual environment
$(VENV_ACTIVATE) : $(PYTHON_VERSION_FILE)
	$(PYENV_PYTHON) -m pip install -U pip virtualenv
	$(PYENV_PYTHON) -m virtualenv $(VENV_DIR)
	$(PYTHON) -m pip install -U pip pip-tools

# set pyenv local Python version
$(PYTHON_VERSION_FILE) : | $(PYENV_PYTHON)
	pyenv local $(PYTHON_VERSION)

# install Python version
$(PYENV_PYTHON) :
	arch -arm64 pyenv install --skip-existing $(PYTHON_VERSION)


# ===== PRE-COMMIT =====
.PHONY : install-pre-commit-hooks
install-pre-commit-hooks : | $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && pre-commit install
