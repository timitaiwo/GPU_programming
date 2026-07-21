# Introduction

This directory details my efforts to deploy and optimize serving of open weight models like Deepseek, Kimi, GLM, MiMO, etc.

I am starting out with [runpod](https://www.runpod.io/) for my GPU infrastructure. As of now (21 July,2026) they exclusively support Nvidia so I'll use that.

1. RTX 5090 - 32 GB VRAM - $0.99/hr
2. H100 SXM - 80GB VRAM - $2.99/hr
3. RTX PRO 6000 - 96GB VRAM - $1.99
4. H200 SXM - 141GB VRAM - $4.39/hr

## Test Run
1. Make sure that `git-lfs` is installed in the system e.g. `sudo dnf install git-lfs` or the equivalent for your OS.

2. I decided to use the [Instella-3B](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html) model from amd. Specifically the [instruct version](https://huggingface.co/amd/Instella-3B-Instruct)
