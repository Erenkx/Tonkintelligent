# Tonkintelligent: An Intelligent Legacy Retrieval System for Tonkin

Tonkin has decades of valuable legacy project data, much of it is siloed, unstructured, and difficult to access. Retrieval often depends on individual memory or internal networks, making knowledge reuse inefficient.

**Tonkintelligent** is an intelligent retrieval system designed to unlock this legacy data. By embedding project documents into a searchable knowledge base, it enables fast, accurate, and reliable information retrieval for all Tonkin stuff.

## Table of Contents

- [Setup Environment](#-setup-environment)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)

## Setup Environment

1. Clone the repository:

```bash
git clone https://github.com/Erenkx/Tonkintelligent.git
cd Tonkintelligent
```

2. Make the installation script executable and run it:

```bash
chmod +x ./install.sh
./install.sh
```

`install.sh` will:

- Automatically check and install Miniconda (if not found)
- Create the Conda environment `tonkintelligent`
- Install all Python dependencies listed in `requirements.txt`

Once finished, your environment is ready to go!

## Data Preparation

Please refer to [Tonkintelligent/data](https://github.com/Erenkx/Tonkintelligent/tree/main/data) for expected folder structure and guidance on preparing your project data.

## Usage

1. Build the systam after preparing your data:

```bash
chmod +x ./build.sh
./build.sh
```

2. Set your OpenAI APY key:
   Copy your OpenAI API key into a plain text file at:

```
code/OPENAI_API_KEY.txt
```

```
DO NOT COMMIT OR PUSH THIS KEY TO THE REPOSITORY!
```

3. Run the system:

```bash
chmod +x ./run.sh
./run.sh
```
