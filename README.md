## facial-recognition_NNclassifier
Utilized PCA + LDA and implemented a Nearest Neighbor Classifier for facial image recognition. Finally, the classification results were visualized to observe the performance of the classifier.
### 使用語言及技術
Python、OpenCV、Scikit-learn、Plotly、電腦視覺、機器學習
### 流程
```mermaid
graph TD
    A[ORL 資料集: 40 類別, 每類 10 筆資料] 
    A --> B[Training Data: 奇數筆資料] 
    A --> C[Testing Data: 偶數筆資料] 

    subgraph PCA + LDA
        D[PCA: n_components=50] --> E[LDA: n_components=20]
    end

    B --> D
    C --> D

    E --> F[Nearest Neighbor Classifier]
    F --> G{Data Visualization: Plotly}

```
### 實作結果
![螢幕擷取畫面 2024-09-23 035916](https://github.com/user-attachments/assets/6a5bf5ff-be49-4fb3-b089-a86439c3f678)

