

## Install Mojo 

Official Guide Here: https://mojolang.org/install/ 

I used pixi so ran `curl -fsSL https://pixi.sh/install.sh | sh` to install.


## Setup
Create the below .vscode/settings.json

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/mojo-gpu-programming/.pixi/envs/default/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true
}
```
