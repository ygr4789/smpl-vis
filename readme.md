## Instruction
Create a micromamba env with the following command
```
micromamba env create -n newenv -f prepare/env.yaml
```

Prepare model files

```
bash prepare/download_smpl_files.sh
```

Executing the file creates output in obj_output
```
python main.py [pkl_file_dir]
```

Run the script in `blender/seq.blend`