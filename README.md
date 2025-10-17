---
title: TDS PROJECT 1
emoji: 📊
colorFrom: green
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: This is an AI app builder and deployer for TDS project 1
---
# AI Web App Builder (Simple)

A lightweight, testable version of an AI-powered web app builder.  
This application accepts task requests, generates web applications using local or LLM-assisted generators, optionally deploys them to GitHub, and notifies an evaluator API. Designed for educational purposes and safe local testing.

---

## Features

- **API Endpoints**:
  - `POST /ready` : Accepts task requests, verifies the secret, schedules background tasks for app generation and deployment.
  - `POST /evaluation-receiver` : Receives evaluation notifications (for testing purposes).
  - `GET /status` : Check the last received task and running background jobs.
  - `GET /health` : Simple health check endpoint.
  - `GET /logs` : Retrieve the latest application logs.

- **Background Job**:
  - Generates files (`index.html`, `README.md`, `LICENSE`) based on task brief.
  - Saves attachments (images, CSV, JSON) from data URIs or URLs.
  - Supports concurrency with a configurable semaphore.

- **Deployment**:
  - Optional GitHub repo creation and push.
  - Optional GitHub Pages deployment.
  - Supports a dry-run mode for local testing without network calls.

- **LLM Integration**:
  - Can call Aipipe/OpenRouter to generate production-ready HTML, README, and LICENSE files.
  - Handles attachments intelligently: images, text files, and metadata.

---

## Table of Contents

- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Task Workflow](#task-workflow)
- [Attachments](#attachments)
- [GitHub Integration](#github-integration)
- [LLM Integration](#llm-integration)
- [Logging](#logging)
- [Notes](#notes)
- [License](#license)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
