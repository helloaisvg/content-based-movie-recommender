# Content-Based Filtering for Movie Recommendations

## Overview
This project develops a neural network-based content-based recommender system using the MovieLens dataset. It generates user and movie embeddings to predict ratings and recommend similar movies. Built with TensorFlow/Keras, it demonstrates skills in deep learning, data preprocessing, and recommendation systems.

## Features
- Neural networks for user and movie feature vectors.
- Predictions for new and existing users.
- Similarity-based recommendations using squared distance.
- Data scaling with StandardScaler and MinMaxScaler.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy, Pandas, Scikit-learn
- Install via: `pip install tensorflow numpy pandas scikit-learn`

## Usage
1. 可选：将 MovieLens `ml-latest-small` 解压至 `./data/ml-latest-small/`（若缺失则自动使用合成数据）。
2. 运行 Notebook：`jupyter notebook content_based_recommender.ipynb`（按顺序运行各单元）。
3. 或直接运行脚本：`python main.py`。
4. 训练完成后在 Notebook 中查看推荐结果示例。

## Results
- Achieved low MSE loss on test set.
- Accurate predictions within 0.5-1.0 of actual ratings.
- Successful similar movie recommendations (e.g., fantasy genre clustering).

## License
MIT License

