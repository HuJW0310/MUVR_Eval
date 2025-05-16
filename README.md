# MUVR
Our Huggingface URL[https://huggingface.co/datasets/debby0527/MUVR]

## Envrironment
```
pip install -r requirments.txt
```
Remember to modify the paths from the code according to your local file structure, the paths in the .sh scripts in particular.

## Prepare Data
```
huggingface-cli download --repo-type dataset --resume-download debby0527/MUVR --local-dir ./MUVR
```

## Run Evaluation

### Run base setting
./scripts/2_muvr_base.sh is a script for the MUVR evaluation under `base` setting.
```
cd ./scripts
bash 2_muvr_base.sh
```
### Run filter setting
./scripts/3_muvr_filter.sh is a script for the MUVR evaluation under `filter` setting.
```
bash 3_muvr_filter.sh
```
### For evaluation on InternVL
Please move the files under `InternVL` to the repo of InternVL and follow their instructions.
The URL of InternVL[https://github.com/OpenGVLab/InternVL].

### For extra feature extraction
We opensourced 2 types of video features we extracted. If extra feature needed, see `./scripts/1_extract_features.sh`
