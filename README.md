# Kaggle competition repository template

This is a template repository aimed at providing a foundation for quickly starting Kaggle competitions.

## Requirements

This repository is designed to be used on macOS.

It requires having [pyenv](https://github.com/pyenv/pyenv) installed.

In order to use the Kaggle API, [authenticating](https://www.kaggle.com/docs/api#authentication) using a Kaggle API token is required.

## How to configure the template

1. Insert your license in the LICENSE file.
2. Input your project configuration parameters in the Makefile.config file. The competition name will be used to download the data from the Kaggle API. It should correspond to the URL of the competition page on Kaggle. If you don't want to download the data on initialisation of the repository, set LOAD_DATA_ON_INIT to false.
3. Run `gmake init` at the root of this repository to initalise the Python environment and download the data for the competition.
4. If you wish to install packages, add them in the requirements.in file and rerun `gmake init` to generate the compiled requirements and install them.
5. Start a Jupyter server running `gmake jupyter`.
6. You can run flake8 and black to lint your python files using the `gmake lint` command. Formatting and linting are also performed on commit via a pre-commit hook.
