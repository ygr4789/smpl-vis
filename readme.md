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

Render an `.obj` sequence as an animation in Blender. This command generates `video/sample.mp4`. `-l` flag will render low-quality results at high speeds.

blender command in blender 4:
```
blender --background --python blender/seq.py -- output/sample
```

<!-- ```
python blender/seq.py output/sample [-l]
```
You can use a makefile to render all the files in the `data` at once. This is rendered in low quality. You can render your favorite ones separately later with the above command.
```
make
``` -->
