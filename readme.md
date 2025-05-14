## Dependencies
Create a micromamba env with the following command
```
micromamba env create -n newenv -f prepare/env.yaml
```

Prepare model files

```
bash prepare/download_smpl_files.sh
```

## Instruction
This command will create `.obj` files in the `output/sample` folder.
```
python main.py data/sample.pkl
```

blender command in blender 4:
```
blender --background --python blender/seq.py -- output/sample
```
The script accepts the following flags:

- `-t, --target`: Render target mode (default=2)
  - 0: Object only - Renders just the object mesh
  - 1: Input motion - Renders object mesh with input motion
  - 2: Refined motion - Renders object mesh with refined motion

- `-c, --camera`: Camera angle selection (default=-1)
  - -1: Renders from all camera angles
  - 0-11: Renders from specific camera angle (see camera.py for angles)

- `-q, --high`: Enable high quality rendering
  - Uses Cycles renderer
  - Slower but better quality

Example commands

```
blender --background --python blender/seq.py -- output/sample -t 1 -c 0 -q
```

