# MUVR
Our Huggingface URL[https://huggingface.co/datasets/debby0527/MUVR]

## Envrironment
```
pip install -r requirments.txt
```

## Prepare Data
```
huggingface-cli download --repo-type dataset --resume-download debby0527/MUVR --local-dir ./MUVR
```

## Run Evaluation

Please modify the paths from the code according to your local file structure, including the paths in the .sh scripts, `./datasets/cuvr.py`, and etc..
Please check the paths relevant to the raw videos and video features with extra attention, the scripts will skip the videos or features if the paths are wrong.

### Run MUVR-Base
`./scripts/2_muvr_base.sh` is a script for `MUVR-Base`.
```
cd ./scripts
bash 2_muvr_base.sh
```
### Run MUVR-Filter
`./scripts/3_muvr_filter.sh` is a script for `MUVR-Filter`.
```
bash 3_muvr_filter.sh
```
### For evaluation on InternVL
Please move the files under `InternVL` to the repo of InternVL for evaluation.
The URL of InternVL[https://github.com/OpenGVLab/InternVL].

### For extra feature extraction
We opensourced 2 types of video features we extracted. If extra feature needed, see `./scripts/1_extract_features.sh`
