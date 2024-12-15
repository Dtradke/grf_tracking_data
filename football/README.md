# Gathering Data from Google Research Football

This repository is adapted from Google Research Football and is used to gather simulated soccer game data.


## Prerequisites

Before starting, ensure you have **Docker** installed on your system. (Note, tested with Version 27.2.0 _

## Getting Started

Follow these steps to set up and run the data gathering script:


### Step 1: Navigate to the Football Directory

Change your working directory to this `football` folder. E.g., from the base project directory `grf_tracking_data/` do:

```bash
cd football/
```

### Step 2: Build and Run the Application in Docker

Use the following command to build and run the application in Docker:

```bash
sudo docker-compose run app
```

### Step 3: Access the GFootball Directory (Inside Docker)

Once inside the Docker container, navigate to the `gfootball` directory:

```bash
cd gfootball
```

### Step 4: Run the Game(s)

To start the games, execute the following command:

```bash
python3 run_games.py
```


Links to Google Research Football:

* [Google Research Football Repository](https://github.com/google-research/football)
* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)


