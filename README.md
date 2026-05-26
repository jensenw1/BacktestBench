This repository is currently under active development. We are continuously working on improving the visual presentation and optimizing the code. If you have any questions, please feel free to open an issue.

## :boom: This work has been accepted for publication in KDD-2026.

## Overall framework of the AutoBacktest.

![Overall framework of the AutoBacktest](figures/WorkFlow_NEW.png)

*This figure is currently being further optimized.*

---

## Overview

This project provides a script to run the model service with configurable parameters. Follow the steps below to set up the environment, install dependencies, start the necessary database container, and execute the script.

---

## Prerequisites

* [Conda](https://docs.conda.io/en/latest/) installed on your system.
* A valid API key for accessing the model backend service.
* [Docker](https://docs.docker.com/) installed and running.

---

## Database Setup

Before running the prediction script, you need to start a PostgreSQL container using Docker and load all the necessary tables into it.

1. **Start the PostgreSQL container**:

   ```bash
   docker run -id \
     --name=quant \
     -v ./data:/var/lib/postgresql/data \
     -p 5432:5432 \
     -e POSTGRES_PASSWORD='123456' \
     -e POSTGRES_USER='quant' \
     -e LANG=C.UTF-8 \
     --restart=always \
     postgres:alpine
   ```
After successful execution, the database will be accessible via port 5432.

2. **Import all tables into the database**:

   The automation script for importing tables is provided as a Python notebook. Run it to populate the database:

   ```bash
   tables/import_tables.ipynb
   ```

---

## Installation

1. Create and activate a Conda environment:

   ```bash
   conda create -n quant python=3.11.14  -y
   conda activate quant
   ```

2. Install required Python packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---

## Reproduction

To reproduce the results, run the following command:

```bash
bash run.sh MODEL_NAME WORKERS BASE_URL API_KEY
```

**Parameter order is important and must be provided exactly as shown above.**

### Parameters

1. **MODEL\_NAME**: The name or identifier of the model you want to serve.

   * Example: `qwen3-8b`, `qwen3-235b`

2. **WORKERS**: The number of worker processes to spawn for handling requests.

   * Example: `4`, `8`

3. **BASE\_URL**: The base URL of the model backend service.

   * Example: `http://localhost:8000`, `https://api.yourmodel.com`

4. **API\_KEY**: Your API key for authenticating with the model backend.

   * Example: `sk-XXXXXXXXXXXXXXXXXXXX`

---

## Example

```bash
bash run.sh qwen3-235b 4 http://localhost:8000 sk-1234567890abcdef
```

This will start the prediction using the `qwen3-235b` model with 4 worker processes, connecting to `http://localhost:8000` using the provided API key.





# Citation
If you found this work useful for you, please consider citing it.
```
@inproceedings{
kdd2026backtestbench,
title={{BacktestBench}: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting},
author={Zhensheng Wang and Wenmian Yang and Qingtai Wu and Lequan Ma and Yiquan Zhang and Weijia Jia},
booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026), August 9--13, 2026, Jeju Island, Republic of Korea},
year={2026},
isbn = {979-8-4007-2259-2/2026/08},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi={10.1145/3770855.3817460},
url={https://arxiv.org/abs/2605.17937}
}
```