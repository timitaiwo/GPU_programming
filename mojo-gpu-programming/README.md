

## Install Mojo 

Official Guide Here: https://mojolang.org/install/ 

I used pixi so ran `curl -fsSL https://pixi.sh/install.sh | sh` to install.

Then ran `pixi create mojo-gpu-programming` in the directory that I want to create project

Run `cd mojo-gpu-programming` to change into the newly created directory

Then ran `pixi add mojo`

To activate the environment run `pixi shell`

### Setup
Create the below .vscode/settings.json if you install the Mojo language extension in vscode

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/mojo-gpu-programming/.pixi/envs/default/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true
}
```
