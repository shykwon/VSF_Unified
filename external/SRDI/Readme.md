# Running the model

Datasets - METR-LA, SOLAR, TRAFFIC, ECG. This code provides a running example with all components on [MTGNN](https://github.com/nnzhan/MTGNN) model (we acknowledge the authors of the work).

## Requirements

The model is implemented using Python3 with dependencies specified in requirements.txt

Meta learning framework using [learn2learn](https://github.com/learnables/learn2learn).

## Data Preparation

### Multivariate time series datasets

Download Solar and Traffic datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.

Download the METR-LA dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git). Move them into the data folder. (Optinally - download the adjacency matrix for META-LA from [here](https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl) and put it as ./data/sensor_graph/adj_mx.pkl , as shown below):

```
wget https://github.com/nnzhan/MTGNN/blob/master/data/sensor_graph/adj_mx.pkl
mkdir data/sensor_graph
mv adj_mx.pkl data/sensor_graph/
```



Download the ECG5000 dataset from [time series classification](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

```
# Create data directories
mkdir -p data/{METR-LA,SOLAR,TRAFFIC,ECG}

# for any dataset, run the following command
python generate_training_data.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```



Here
{0} is for the dataset: metr-la, solar, traffic, ECG
{1} is the directory where to save the train, valid, test splits. These are created from the first command
{2} the raw data filename (the downloaded file), such as - ECG_data.csv, metr-la.hd5, solar.txt, traffic.txt

## Training

```
python train_multi_step.py --data ./data/{0} --model_name {1} --device cuda:0 --expid {2} --epochs 100 --batch_size 64 --runs 10 --random_node_idx_split_runs 100 --lower_limit_random_node_selections 15 --upper_limit_random_node_selections 15 --step_size1 {3} --mask_remaining 
```

Here,
{0} - refers to the dataset directory: ./data/{ECG/TRAFFIC/METR-LA/SOLAR}
{1} - refers to the model name
{2} - refers to the manually assigned "ID" of the experiment
{3} - step_size1 is 2500 for METR-LA and SOLAR, 400 for ECG, 1000 for TRAFFIC

## Note

If there are multiple CUDAs, please manually change the CUDA device in model. py, main_madel. py, diff_madel. py